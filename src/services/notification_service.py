"""
XYZ Analytics - Notification Service

This module handles the notification subsystem for the Conversation Analytics Platform.
It monitors processed conversations for pattern matches and dispatches alerts via
various channels (email, webhook, in-app).

Note: This is a simplified version for assessment purposes.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Any, Optional, Callable


class NotificationChannel(Enum):
    EMAIL = "email"
    WEBHOOK = "webhook"
    IN_APP = "in_app"


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class TriggerOperator(Enum):
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN_OR_EQUAL = "gte"
    LESS_THAN_OR_EQUAL = "lte"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"


@dataclass
class NotificationRule:
    """
    Defines a rule for triggering notifications based on conversation analysis.
    """
    id: str
    name: str
    customer_id: str
    enabled: bool
    metric: str  # e.g., "qa_score", "sentiment", "compliance_flags"
    operator: TriggerOperator
    threshold: Any
    time_window_minutes: int  # Aggregation window
    min_occurrences: int  # Minimum matches before triggering
    channels: List[NotificationChannel]
    severity: AlertSeverity
    cooldown_minutes: int = 60  # Minimum time between notifications
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_triggered_at: Optional[datetime] = None


@dataclass 
class NotificationEvent:
    """
    Represents a triggered notification that should be dispatched.
    """
    id: str
    rule_id: str
    customer_id: str
    severity: AlertSeverity
    channels: List[NotificationChannel]
    title: str
    message: str
    data: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.utcnow)
    dispatched_at: Optional[datetime] = None
    dispatch_status: Dict[str, str] = field(default_factory=dict)


class NotificationService:
    """
    Core notification service responsible for:
    - Managing notification rules
    - Evaluating incoming data against rules
    - Tracking state for time-window based alerting
    - Dispatching notifications via configured channels
    """
    
    def __init__(self):
        self.rules: Dict[str, NotificationRule] = {}
        self.events: List[NotificationEvent] = []
        self.state_store: Dict[str, List[Dict]] = {}  # Rule ID -> matching events
        self.dispatch_handlers: Dict[NotificationChannel, Callable] = {}
        
        # Register default handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default notification dispatch handlers"""
        self.dispatch_handlers[NotificationChannel.EMAIL] = self._send_email
        self.dispatch_handlers[NotificationChannel.WEBHOOK] = self._send_webhook
        self.dispatch_handlers[NotificationChannel.IN_APP] = self._send_in_app
    
    def create_rule(self, rule: NotificationRule) -> NotificationRule:
        """
        Create a new notification rule.
        
        Raises ValueError if validation fails.
        """
        self._validate_rule(rule)
        self.rules[rule.id] = rule
        self.state_store[rule.id] = []
        return rule
    
    def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> NotificationRule:
        """
        Update an existing rule.
        
        BUG: There's an issue with how updates are applied - can you find it?
        """
        if rule_id not in self.rules:
            raise ValueError(f"Rule not found: {rule_id}")
        
        rule = self.rules[rule_id]
        
        for key, value in updates.items():
            if hasattr(rule, key):
                setattr(rule, key, value)
        
        # Re-validate after updates
        self._validate_rule(rule)
        
        return rule
    
    def delete_rule(self, rule_id: str, customer_id: str) -> bool:
        """
        Delete a notification rule.
        
        BUG: There's a security issue here - can you identify it?
        """
        if rule_id not in self.rules:
            return False
        
        del self.rules[rule_id]
        if rule_id in self.state_store:
            del self.state_store[rule_id]
        
        return True
    
    def get_rules(self, customer_id: str) -> List[NotificationRule]:
        """Get all rules for a customer"""
        return [r for r in self.rules.values() if r.customer_id == customer_id]
    
    def _validate_rule(self, rule: NotificationRule) -> None:
        """
        Validate a notification rule.
        
        BUG: The validation is incomplete - what's missing?
        """
        if not rule.name or len(rule.name.strip()) == 0:
            raise ValueError("Rule name cannot be empty")
        
        if rule.time_window_minutes < 1:
            raise ValueError("Time window must be at least 1 minute")
        
        if rule.min_occurrences < 1:
            raise ValueError("Minimum occurrences must be at least 1")
        
        if len(rule.channels) == 0:
            raise ValueError("At least one notification channel is required")
        
        # Validate metric-specific constraints
        valid_metrics = ["qa_score", "sentiment", "compliance_flags", "topic_match", "keyword_match"]
        if rule.metric not in valid_metrics:
            raise ValueError(f"Invalid metric: {rule.metric}")
    
    def evaluate(self, conversation_id: str, customer_id: str, analysis: Dict[str, Any]) -> List[NotificationEvent]:
        """
        Evaluate analysis results against all applicable rules.
        
        This is the main entry point called after a conversation is processed.
        
        Returns a list of triggered notification events.
        """
        triggered_events = []
        now = datetime.utcnow()
        
        # Get all enabled rules for this customer
        applicable_rules = [
            r for r in self.rules.values() 
            if r.customer_id == customer_id and r.enabled
        ]
        
        for rule in applicable_rules:
            # Check cooldown
            if rule.last_triggered_at:
                cooldown_end = rule.last_triggered_at + timedelta(minutes=rule.cooldown_minutes)
                if now < cooldown_end:
                    continue
            
            # Evaluate the rule condition
            if self._evaluate_condition(rule, analysis):
                # Add to state store for this rule
                self.state_store[rule.id].append({
                    "conversation_id": conversation_id,
                    "timestamp": now,
                    "analysis": analysis
                })
                
                # Clean up old entries outside the time window
                window_start = now - timedelta(minutes=rule.time_window_minutes)
                self.state_store[rule.id] = [
                    e for e in self.state_store[rule.id]
                    if e["timestamp"] >= window_start
                ]
                
                # Check if we've reached the threshold
                if len(self.state_store[rule.id]) >= rule.min_occurrences:
                    event = self._create_event(rule, self.state_store[rule.id])
                    triggered_events.append(event)
                    rule.last_triggered_at = now
                    self.state_store[rule.id] = []  # Reset state
        
        return triggered_events
    
    def _evaluate_condition(self, rule: NotificationRule, analysis: Dict[str, Any]) -> bool:
        """
        Evaluate whether the analysis matches the rule condition.
        
        BUG: There's an edge case bug in the comparison logic. Can you find it?
        """
        value = analysis.get(rule.metric)
        
        if value is None:
            return False
        
        threshold = rule.threshold
        
        if rule.operator == TriggerOperator.GREATER_THAN:
            return value > threshold
        elif rule.operator == TriggerOperator.LESS_THAN:
            return value < threshold
        elif rule.operator == TriggerOperator.EQUALS:
            return value == threshold
        elif rule.operator == TriggerOperator.NOT_EQUALS:
            return value != threshold
        elif rule.operator == TriggerOperator.GREATER_THAN_OR_EQUAL:
            return value >= threshold
        elif rule.operator == TriggerOperator.LESS_THAN_OR_EQUAL:
            return value <= threshold
        elif rule.operator == TriggerOperator.CONTAINS:
            if isinstance(value, list):
                return threshold in value
            return threshold in str(value)
        elif rule.operator == TriggerOperator.NOT_CONTAINS:
            if isinstance(value, list):
                return threshold not in value
            return threshold not in str(value)
        
        return False
    
    def _create_event(self, rule: NotificationRule, matching_data: List[Dict]) -> NotificationEvent:
        """Create a notification event from a triggered rule"""
        event_id = hashlib.md5(
            f"{rule.id}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]
        
        conversation_ids = [d["conversation_id"] for d in matching_data]
        
        title = f"Alert: {rule.name}"
        message = (
            f"Rule '{rule.name}' triggered.\n"
            f"Metric: {rule.metric}\n"
            f"Condition: {rule.operator.value} {rule.threshold}\n"
            f"Occurrences: {len(matching_data)} in the last {rule.time_window_minutes} minutes\n"
            f"Affected conversations: {', '.join(conversation_ids)}"
        )
        
        event = NotificationEvent(
            id=event_id,
            rule_id=rule.id,
            customer_id=rule.customer_id,
            severity=rule.severity,
            channels=rule.channels,
            title=title,
            message=message,
            data={
                "rule_name": rule.name,
                "metric": rule.metric,
                "threshold": rule.threshold,
                "conversation_ids": conversation_ids,
                "match_count": len(matching_data)
            }
        )
        
        self.events.append(event)
        return event
    
    def dispatch(self, event: NotificationEvent) -> Dict[str, str]:
        """
        Dispatch a notification event to all configured channels.
        
        Returns a dict of channel -> status for each dispatch attempt.
        """
        results = {}
        
        for channel in event.channels:
            handler = self.dispatch_handlers.get(channel)
            if not handler:
                results[channel.value] = "error: no handler"
                continue
            
            try:
                handler(event)
                results[channel.value] = "success"
            except Exception as e:
                results[channel.value] = f"error: {str(e)}"
        
        event.dispatched_at = datetime.utcnow()
        event.dispatch_status = results
        
        return results
    
    def _send_email(self, event: NotificationEvent) -> None:
        """
        Send notification via email.
        
        In production, this would integrate with SendGrid or similar.
        For testing purposes, this is a mock implementation.
        """
        # Mock implementation - would integrate with email service
        print(f"[EMAIL] Sending notification to customer {event.customer_id}: {event.title}")
    
    def _send_webhook(self, event: NotificationEvent) -> None:
        """
        Send notification via webhook.
        
        In production, this would POST to configured webhook URLs.
        
        BUG: There's a potential issue with the payload construction.
        """
        payload = {
            "event_id": event.id,
            "timestamp": event.created_at.isoformat(),
            "severity": event.severity.value,
            "title": event.title,
            "message": event.message,
            "data": event.data
        }
        
        # Mock implementation - would POST to webhook URL
        print(f"[WEBHOOK] Sending payload: {json.dumps(payload)}")
    
    def _send_in_app(self, event: NotificationEvent) -> None:
        """
        Send in-app notification.
        
        In production, this would push to a real-time notification system.
        """
        # Mock implementation
        print(f"[IN-APP] Notification for {event.customer_id}: {event.title}")
    
    def get_events(
        self, 
        customer_id: str,
        severity: Optional[AlertSeverity] = None,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[NotificationEvent]:
        """
        Retrieve notification events for a customer.
        
        BUG: There's an issue with the filtering logic.
        """
        results = []
        
        for event in self.events:
            if event.customer_id != customer_id:
                continue
            
            if severity and event.severity != severity:
                continue
            
            if since and event.created_at < since:
                continue
            
            results.append(event)
        
        # Sort by created_at descending
        results.sort(key=lambda x: x.created_at, reverse=True)
        
        return results[:limit]


class RuleBuilder:
    """
    Fluent builder for creating notification rules.
    
    Example:
        rule = RuleBuilder("cust_001")\
            .with_name("Low QA Score Alert")\
            .when_metric("qa_score")\
            .is_less_than(60.0)\
            .within_minutes(30)\
            .minimum_occurrences(3)\
            .notify_via([NotificationChannel.EMAIL])\
            .with_severity(AlertSeverity.WARNING)\
            .build()
    """
    
    def __init__(self, customer_id: str):
        self._customer_id = customer_id
        self._name: Optional[str] = None
        self._metric: Optional[str] = None
        self._operator: Optional[TriggerOperator] = None
        self._threshold: Any = None
        self._time_window: int = 60
        self._min_occurrences: int = 1
        self._channels: List[NotificationChannel] = []
        self._severity: AlertSeverity = AlertSeverity.INFO
        self._cooldown: int = 60
        self._enabled: bool = True
        self._validators = []
    
    def with_name(self, name: str) -> "RuleBuilder":
        self._name = name
        return self
    
    def when_metric(self, metric: str) -> "RuleBuilder":
        self._metric = metric
        return self
    
    def is_greater_than(self, value: Any) -> "RuleBuilder":
        self._operator = TriggerOperator.GREATER_THAN
        self._threshold = value
        return self
    
    def is_less_than(self, value: Any) -> "RuleBuilder":
        self._operator = TriggerOperator.LESS_THAN
        self._threshold = value
        return self
    
    def equals(self, value: Any) -> "RuleBuilder":
        self._operator = TriggerOperator.EQUALS
        self._threshold = value
        return self
    
    def contains(self, value: Any) -> "RuleBuilder":
        self._operator = TriggerOperator.CONTAINS
        self._threshold = value
        return self
    
    def within_minutes(self, minutes: int) -> "RuleBuilder":
        self._time_window = minutes
        return self
    
    def minimum_occurrences(self, count: int) -> "RuleBuilder":
        self._min_occurrences = count
        return self
    
    def notify_via(self, channels: List[NotificationChannel]) -> "RuleBuilder":
        self._channels = channels
        return self
    
    def with_severity(self, severity: AlertSeverity) -> "RuleBuilder":
        self._severity = severity
        return self
    
    def with_cooldown(self, minutes: int) -> "RuleBuilder":
        self._cooldown = minutes
        return self
    
    def disabled(self) -> "RuleBuilder":
        self._enabled = False
        return self
    
    def build(self) -> NotificationRule:
        """Build the notification rule"""
        if not self._name:
            raise ValueError("Rule name is required")
        if not self._metric:
            raise ValueError("Metric is required")
        if self._operator is None:
            raise ValueError("Operator is required")
        if len(self._channels) == 0:
            raise ValueError("At least one notification channel is required")
        
        rule_id = hashlib.md5(
            f"{self._customer_id}_{self._name}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:12]
        
        return NotificationRule(
            id=rule_id,
            name=self._name,
            customer_id=self._customer_id,
            enabled=self._enabled,
            metric=self._metric,
            operator=self._operator,
            threshold=self._threshold,
            time_window_minutes=self._time_window,
            min_occurrences=self._min_occurrences,
            channels=self._channels,
            severity=self._severity,
            cooldown_minutes=self._cooldown
        )

    def is_equal_to(self, param):
        self.operator = "EQUALS"
        self.value = param
        return self

