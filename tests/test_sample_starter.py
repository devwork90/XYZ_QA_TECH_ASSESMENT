"""
XYZ Analytics - Sample Test Suite

This file contains starter tests to help you understand the testing patterns
and frameworks expected for this assessment.

Your task is to significantly expand upon these tests as outlined in the
assessment instructions.
"""

import json
import os
from datetime import datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from websockets.sync import router

# Imports from the src modules (path configured in conftest.py)
from src.api.conversation_api import (
    ConversationCreate,
    ConversationResponse,
    Speaker,
    Utterance,
    ConversationType,
    ConversationStatus,
    data_store,
    validate_api_key,
)
from src.services.notification_service import (
    NotificationService,
    NotificationRule,
    NotificationChannel,
    AlertSeverity,
    TriggerOperator,
    RuleBuilder,
)
from src.services.qa_scoring_engine import (
    QAScoringEngine,
    ScoreCategory,
    ComplianceFlag,
)

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from src.main import app
    return TestClient(app)

@pytest.fixture
def sample_speakers():
    """Sample speaker list for testing"""
    return [
        Speaker(id="agent_1", role="agent", name="Test Agent"),
        Speaker(id="customer_1", role="customer", name="Test Customer"),
    ]


@pytest.fixture
def sample_utterances():
    """Sample utterances for testing"""
    return [
        Utterance(
            speaker_id="agent_1",
            text="Hello, thank you for calling support.",
            start_time=0.0,
            end_time=3.0,
            confidence=0.95,
        ),
        Utterance(
            speaker_id="customer_1",
            text="Hi, I need help with my account.",
            start_time=3.5,
            end_time=6.0,
            confidence=0.92,
        ),
    ]


@pytest.fixture
def valid_conversation_data(sample_speakers, sample_utterances):
    """Valid conversation payload for testing"""
    return ConversationCreate(
        external_id="test_external_001",
        conversation_type=ConversationType.CALL,
        speakers=sample_speakers,
        utterances=sample_utterances,
        metadata={"source": "test"},
        recorded_at=datetime(2024, 1, 15, 10, 0),
        duration_seconds=120.5,
        language="en",
    )

@pytest.fixture
def valid_speakers():
    return [
        Speaker(id="spk_1", role="agent", name="Agent Smith"),
        Speaker(id="spk_2", role="customer", name="John Doe"),
    ]

@pytest.fixture
def valid_utterances():
    return [
        Utterance(
            speaker_id="spk_1",
            text="Hello",
            start_time=0.0,
            end_time=1.0,
            confidence=0.95,
        ),
        Utterance(
            speaker_id="spk_2",
            text="Hi",
            start_time=1.1,
            end_time=2.0,
            confidence=0.90,
        ),
    ]


@pytest.fixture
def base_conversation_payload(valid_speakers, valid_utterances):
    return dict(
        external_id="ext_123",
        conversation_type=ConversationType.CALL,
        speakers=valid_speakers,
        utterances=valid_utterances,
        metadata={},
        recorded_at=datetime.utcnow(),
        duration_seconds=120.0,
        language="en",
    )

@pytest.fixture
def notification_service():
    """Fresh NotificationService instance"""
    return NotificationService()


@pytest.fixture
def qa_engine():
    """Fresh QAScoringEngine instance"""
    return QAScoringEngine()


@pytest.fixture
def load_test_fixtures():
    """Load test fixtures from JSON file"""
    fixtures_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "test_fixtures.json"
    )
    with open(fixtures_path, "r") as f:
        return json.load(f)


# =============================================================================
# Sample API Tests - Expand these significantly
# =============================================================================


class TestConversationValidation:
    """Tests for conversation input validation"""

    def test_valid_conversation_passes_validation(self, valid_conversation_data):
        """Valid conversation data should pass all validation"""
        # This should not raise any exceptions
        assert valid_conversation_data.external_id == "test_external_001"
        assert len(valid_conversation_data.speakers) == 2
        assert len(valid_conversation_data.utterances) == 2

    def test_empty_external_id_fails_validation(
        self, sample_speakers, sample_utterances
    ):
        """Empty external_id should fail validation"""
        with pytest.raises(ValueError):
            ConversationCreate(
                external_id="",  # Invalid: empty
                conversation_type=ConversationType.CALL,
                speakers=sample_speakers,
                utterances=sample_utterances,
                recorded_at=datetime.now(),
                duration_seconds=60.0,
            )

    def test_duplicate_speaker_ids_fails_validation(self, sample_utterances):
        """Duplicate speaker IDs should fail validation"""
        duplicate_speakers = [
            Speaker(id="same_id", role="agent", name="Agent 1"),
            Speaker(id="same_id", role="customer", name="Customer 1"),  # Duplicate ID
        ]

        with pytest.raises(ValueError) as exc_info:
            ConversationCreate(
                external_id="test_001",
                conversation_type=ConversationType.CALL,
                speakers=duplicate_speakers,
                utterances=sample_utterances,
                recorded_at=datetime.now(),
                duration_seconds=60.0,
            )

        assert "unique" in str(exc_info.value).lower()

    # TODO: Add more validation tests
    # - Test utterance end_time > start_time validation
    # - Test speaker_id references in utterances
    # - Test boundary values for duration_seconds
    # - Test language code format

    # ============================================================
    # TODO Tests Completed
    # ============================================================

    # ------------------------------------------------------------
    # 1️ Test utterance end_time > start_time validation
    # ------------------------------------------------------------

    def test_utterance_end_time_less_than_start_time(self):
        with pytest.raises(ValidationError) as exc_info:
            Utterance(
                speaker_id="spk_1",
                text="Invalid timing",
                start_time=5.0,
                end_time=4.0,  # invalid
            )

        assert "end_time must be greater than or equal to start_time" in str(exc_info.value)

    # ------------------------------------------------------------
    # 2️ Test speaker_id references in utterances
    # ------------------------------------------------------------

    def test_invalid_speaker_reference(self, valid_speakers, base_conversation_payload):
        invalid_utterances = [
            Utterance(
                speaker_id="INVALID_SPK",  # not in speakers
                text="Hello",
                start_time=0.0,
                end_time=1.0,
            )
        ]

        payload = base_conversation_payload.copy()
        payload["utterances"] = invalid_utterances

        with pytest.raises(ValidationError) as exc_info:
            ConversationCreate(**payload)

        assert "Invalid speaker_id" in str(exc_info.value)

    # ------------------------------------------------------------
    # 3️ Test boundary values for duration_seconds
    #     Field(..., gt=0, le=86400)
    # ------------------------------------------------------------

    def test_duration_seconds_zero_invalid(self, base_conversation_payload):
        payload = base_conversation_payload.copy()
        payload["duration_seconds"] = 0  # invalid (gt=0)

        with pytest.raises(ValidationError):
            ConversationCreate(**payload)

    def test_duration_seconds_max_boundary_valid(self, base_conversation_payload):
        payload = base_conversation_payload.copy()
        payload["duration_seconds"] = 86400  # valid max boundary

        model = ConversationCreate(**payload)
        assert model.duration_seconds == 86400

    def test_duration_seconds_above_max_invalid(self, base_conversation_payload):
        payload = base_conversation_payload.copy()
        payload["duration_seconds"] = 86401  # invalid (> 86400)

        with pytest.raises(ValidationError):
            ConversationCreate(**payload)

    # ------------------------------------------------------------
    # 4️ Test language code format
    #     pattern="^[a-z]{2}$"
    # ------------------------------------------------------------

    def test_valid_language_code(self, base_conversation_payload):
        payload = base_conversation_payload.copy()
        payload["language"] = "fr"

        model = ConversationCreate(**payload)
        assert model.language == "fr"

    def test_invalid_language_uppercase(self, base_conversation_payload):
        payload = base_conversation_payload.copy()
        payload["language"] = "EN"  # invalid (must be lowercase)

        with pytest.raises(ValidationError):
            ConversationCreate(**payload)

    def test_invalid_language_three_letters(self, base_conversation_payload):
        payload = base_conversation_payload.copy()
        payload["language"] = "eng"  # invalid (3 letters)

        with pytest.raises(ValidationError):
            ConversationCreate(**payload)

    def test_invalid_language_numeric(self, base_conversation_payload):
        payload = base_conversation_payload.copy()
        payload["language"] = "e1"  # invalid (must be letters only)

        with pytest.raises(ValidationError):
            ConversationCreate(**payload)


class TestUtteranceValidation:
    """Tests for utterance-level validation"""

    def test_valid_utterance(self):
        """Valid utterance should be created successfully"""
        utt = Utterance(
            speaker_id="agent_1",
            text="Hello, how can I help?",
            start_time=0.0,
            end_time=2.5,
            confidence=0.95,
        )
        assert utt.text == "Hello, how can I help?"
        assert utt.confidence == 0.95

    def test_end_time_before_start_time_fails(self):
        """end_time before start_time should fail validation"""
        with pytest.raises(ValueError):
            Utterance(
                speaker_id="agent_1",
                text="Test",
                start_time=5.0,
                end_time=2.0,  # Invalid: before start_time
                confidence=0.9,
            )

    # TODO: Add more utterance validation tests
    def test_end_time_negative_fails(self):
        with pytest.raises(ValidationError):
            Utterance(
                speaker_id="agent_1",
                text="Test",
                start_time=0.0,
                end_time=-5.0,  # Invalid
                confidence=0.9,
            )

    # ------------------------------------------------------------
    # Text validation
    # ------------------------------------------------------------

    def test_empty_text_fails(self):
        with pytest.raises(ValidationError):
            Utterance(
                speaker_id="agent_1",
                text="",  # min_length=1
                start_time=0.0,
                end_time=1.0,
            )

    def test_text_too_long_fails(self):
        long_text = "a" * 10001  # max_length=10000

        with pytest.raises(ValidationError):
            Utterance(
                speaker_id="agent_1",
                text=long_text,
                start_time=0.0,
                end_time=1.0,
            )

    # ------------------------------------------------------------
    # Confidence validation
    # ------------------------------------------------------------

    def test_confidence_below_zero_fails(self):
        with pytest.raises(ValidationError):
            Utterance(
                speaker_id="agent_1",
                text="Test",
                start_time=0.0,
                end_time=1.0,
                confidence=-0.1,  # Invalid
            )

    def test_confidence_above_one_fails(self):
        with pytest.raises(ValidationError):
            Utterance(
                speaker_id="agent_1",
                text="Test",
                start_time=0.0,
                end_time=1.0,
                confidence=1.5,  # Invalid
            )

    def test_confidence_zero_valid(self):
        utt = Utterance(
            speaker_id="agent_1",
            text="Test",
            start_time=0.0,
            end_time=1.0,
            confidence=0.0,  # Boundary valid
        )
        assert utt.confidence == 0.0

    def test_confidence_one_valid(self):
        utt = Utterance(
            speaker_id="agent_1",
            text="Test",
            start_time=0.0,
            end_time=1.0,
            confidence=1.0,  # Boundary valid
        )
        assert utt.confidence == 1.0

    # ------------------------------------------------------------
    # Boundary timing case
    # ------------------------------------------------------------

    def test_start_time_equals_end_time_valid(self):
        """
        end_time == start_time is allowed by validator
        (>= comparison)
        """
        utt = Utterance(
            speaker_id="agent_1",
            text="Instant utterance",
            start_time=5.0,
            end_time=5.0,
            confidence=0.8,
        )

        assert utt.end_time == utt.start_time

    # ------------------------------------------------------------
    # Required field validation
    # ------------------------------------------------------------

    def test_missing_speaker_id_fails(self):
        with pytest.raises(ValidationError):
            Utterance(
                text="Test",
                start_time=0.0,
                end_time=1.0,
            )


# =============================================================================
# Sample Notification Service Tests - Expand these significantly
# =============================================================================


class TestNotificationRuleCreation:
    """Tests for notification rule creation"""

    def test_create_valid_rule(self, notification_service):
        """Should successfully create a valid rule"""
        rule = (
            RuleBuilder("cust_001")
            .with_name("Test Rule")
            .when_metric("qa_score")
            .is_less_than(50.0)
            .within_minutes(30)
            .minimum_occurrences(1)
            .notify_via([NotificationChannel.EMAIL])
            .with_severity(AlertSeverity.WARNING)
            .build()
        )

        created = notification_service.create_rule(rule)

        assert created.name == "Test Rule"
        assert created.customer_id == "cust_001"
        assert created.metric == "qa_score"
        assert created.threshold == 50.0

    def test_rule_without_name_fails(self, notification_service):
        """Rule without name should fail validation"""
        with pytest.raises(ValueError):
            RuleBuilder("cust_001").when_metric("qa_score").is_less_than(
                50.0
            ).notify_via([NotificationChannel.EMAIL]).build()

    # TODO: Add more rule creation tests
    # - Test invalid metrics
    # - Test missing channels
    # - Test boundary values for time windows
# ------------------------------------------------------------
# 1️ Invalid Metrics
# ------------------------------------------------------------

def test_invalid_metric_name_fails(notification_service):
    """Using unsupported metric should fail"""
    with pytest.raises(ValueError):
        (
            RuleBuilder("cust_001")
            .with_name("Low Score Alert")
            .when_metric("invalid_metric")  # invalid
            .is_less_than(50.0)
            .notify_via([NotificationChannel.EMAIL])
            .build()
        )


def test_missing_metric_operator_fails(notification_service):
    """Metric defined but operator not set should fail"""
    with pytest.raises(ValueError):
        (
            RuleBuilder("cust_001")
            .with_name("Incomplete Rule")
            .when_metric("qa_score")
            # Missing .is_less_than / is_greater_than
            .notify_via([NotificationChannel.EMAIL])
            .build()
        )


# ------------------------------------------------------------
# 2️ Missing Notification Channels
# ------------------------------------------------------------

def test_missing_notification_channels_fails(notification_service):
    """Rule without notify_via should fail"""
    with pytest.raises(ValueError):
        (
            RuleBuilder("cust_001")
            .with_name("No Channels Rule")
            .when_metric("qa_score")
            .is_less_than(50.0)
            # Missing notify_via
            .build()
        )


def test_empty_notification_channels_fails(notification_service):
    """Empty notification channel list should fail"""
    with pytest.raises(ValueError):
        (
            RuleBuilder("cust_001")
            .with_name("Empty Channels Rule")
            .when_metric("qa_score")
            .is_less_than(50.0)
            .notify_via([])  # Invalid
            .build()
        )


# ------------------------------------------------------------
# 3️ Boundary Values for Time Windows
# ------------------------------------------------------------

def test_zero_time_window_fails(notification_service):
    """Zero minute time window should fail"""
    with pytest.raises(ValueError):
        (
            RuleBuilder("cust_001")
            .with_name("Zero Window Rule")
            .when_metric("qa_score")
            .is_less_than(50.0)
            .within_minutes(0)  # boundary invalid
            .notify_via([NotificationChannel.EMAIL])
            .build()
        )


def test_negative_time_window_fails(notification_service):
    """Negative time window should fail"""
    with pytest.raises(ValueError):
        (
            RuleBuilder("cust_001")
            .with_name("Negative Window Rule")
            .when_metric("qa_score")
            .is_less_than(50.0)
            .within_minutes(-10)
            .notify_via([NotificationChannel.EMAIL])
            .build()
        )


def test_valid_time_window_boundary(notification_service):
    """Smallest valid time window should succeed"""
    rule = (
        RuleBuilder("cust_001")
        .with_name("Valid Boundary Window")
        .when_metric("qa_score")
        .is_less_than(50.0)
        .within_minutes(1)  # boundary valid
        .notify_via([NotificationChannel.EMAIL])
        .build()
    )

    assert rule is not None
    assert rule.time_window_minutes == 1


# ------------------------------------------------------------
# 4️ Metric Boundary Values
# ------------------------------------------------------------

def test_metric_boundary_zero_valid(notification_service):
    """Metric value at 0 boundary should be valid"""
    rule = (
        RuleBuilder("cust_001")
        .with_name("Zero Threshold")
        .when_metric("qa_score")
        .is_less_than(0.0)
        .notify_via([NotificationChannel.EMAIL])
        .build()
    )

    assert rule.threshold == 0.0


def test_metric_above_max_boundary_fails(notification_service):
    """Metric above allowed range (e.g. qa_score > 100) should fail"""
    with pytest.raises(ValueError):
        (
            RuleBuilder("cust_001")
            .with_name("Too High Threshold")
            .when_metric("qa_score")
            .is_greater_than(150.0)  # invalid for qa_score
            .notify_via([NotificationChannel.EMAIL])
            .build()
        )

class TestNotificationRuleEvaluation:
    """Tests for notification rule evaluation"""

    def test_rule_triggers_when_threshold_exceeded(self, notification_service):
        """Rule should trigger when condition is met"""
        rule = (
            RuleBuilder("cust_001")
            .with_name("Low Score Alert")
            .when_metric("qa_score")
            .is_less_than(60.0)
            .within_minutes(60)
            .minimum_occurrences(1)
            .notify_via([NotificationChannel.EMAIL])
            .build()
        )

        notification_service.create_rule(rule)

        # Simulate analysis result that should trigger
        analysis = {"qa_score": 45.0}

        events = notification_service.evaluate("conv_001", "cust_001", analysis)

        assert len(events) == 1
        assert events[0].rule_id == rule.id

    # TODO: Add more evaluation tests
    # - Test rule does NOT trigger when threshold not met
    def test_rule_does_not_trigger_when_threshold_not_met(self, notification_service):
        """Rule should NOT trigger when condition is not met"""
        rule = (
            RuleBuilder("cust_001")
            .with_name("Low Score Alert")
            .when_metric("qa_score")
            .is_less_than(60.0)
            .within_minutes(60)
            .minimum_occurrences(1)
            .notify_via([NotificationChannel.EMAIL])
            .build()
        )

        notification_service.create_rule(rule)

        # Score above threshold → should NOT trigger
        analysis = {"qa_score": 85.0}

        events = notification_service.evaluate("conv_001", "cust_001", analysis)

        assert events == [] or len(events) == 0

    # - Test time window aggregation
    def test_rule_triggers_only_after_minimum_occurrences_within_window(
            self, notification_service
    ):
        """Rule should trigger only after required occurrences within time window"""
        rule = (
            RuleBuilder("cust_001")
            .with_name("Repeated Low Score")
            .when_metric("qa_score")
            .is_less_than(60.0)
            .within_minutes(60)
            .minimum_occurrences(3)
            .notify_via([NotificationChannel.EMAIL])
            .build()
        )

        notification_service.create_rule(rule)

        # First occurrence
        events1 = notification_service.evaluate(
            "conv_001", "cust_001", {"qa_score": 50.0}
        )
        assert len(events1) == 0

        # Second occurrence
        events2 = notification_service.evaluate(
            "conv_001", "cust_001", {"qa_score": 55.0}
        )
        assert len(events2) == 0

        # Third occurrence → should trigger
        events3 = notification_service.evaluate(
            "conv_001", "cust_001", {"qa_score": 40.0}
        )
        assert len(events3) == 1
        assert events3[0].rule_id == rule.id

    # - Test cooldown period
    def test_rule_respects_cooldown_period(self, notification_service):
        """Rule should not trigger again during cooldown period"""
        rule = (
            RuleBuilder("cust_001")
            .with_name("Cooldown Test")
            .when_metric("qa_score")
            .is_less_than(60.0)
            .within_minutes(60)
            .minimum_occurrences(1)
            .with_cooldown_minutes(30)  # assuming this exists
            .notify_via([NotificationChannel.EMAIL])
            .build()
        )

        notification_service.create_rule(rule)

        analysis = {"qa_score": 45.0}

        # First trigger
        events1 = notification_service.evaluate("conv_001", "cust_001", analysis)
        assert len(events1) == 1

        # Immediate second evaluation → should be suppressed
        events2 = notification_service.evaluate("conv_001", "cust_001", analysis)
        assert len(events2) == 0

    # - Test multiple rules evaluating same data
    def test_multiple_rules_can_trigger_from_same_analysis(
            self, notification_service
    ):
        """Multiple rules should independently evaluate the same input"""
        rule1 = (
            RuleBuilder("cust_001")
            .with_name("Low Score Alert")
            .when_metric("qa_score")
            .is_less_than(60.0)
            .within_minutes(60)
            .minimum_occurrences(1)
            .notify_via([NotificationChannel.EMAIL])
            .build()
        )

        rule2 = (
            RuleBuilder("cust_001")
            .with_name("Very Low Score Alert")
            .when_metric("qa_score")
            .is_less_than(50.0)
            .within_minutes(60)
            .minimum_occurrences(1)
            .notify_via([NotificationChannel.EMAIL])
            .build()
        )

        notification_service.create_rule(rule1)
        notification_service.create_rule(rule2)

        # Score satisfies both rules
        analysis = {"qa_score": 45.0}

        events = notification_service.evaluate("conv_001", "cust_001", analysis)

        triggered_rule_ids = {e.rule_id for e in events}

        assert len(events) == 2
        assert rule1.id in triggered_rule_ids
        assert rule2.id in triggered_rule_ids

# =============================================================================
# Sample QA Scoring Tests - Expand these significantly
# =============================================================================


class TestQAScoring:
    """Tests for QA scoring engine"""

    def test_high_quality_conversation_scores_well(self, qa_engine, load_test_fixtures):
        """A well-handled conversation should score highly"""
        conv = load_test_fixtures["sample_conversations"][0]  # Good conversation

        result = qa_engine.score_conversation(
            conversation_id=conv["id"],
            speakers=conv["speakers"],
            utterances=conv["utterances"],
            metadata=conv.get("metadata"),
        )

        assert result.overall_score >= 80.0
        assert len(result.compliance_flags) == 0

    def test_poor_conversation_scores_low(self, qa_engine, load_test_fixtures):
        """A poorly handled conversation should score low"""
        conv = load_test_fixtures["sample_conversations"][1]  # Bad conversation

        result = qa_engine.score_conversation(
            conversation_id=conv["id"],
            speakers=conv["speakers"],
            utterances=conv["utterances"],
            metadata=conv.get("metadata"),
        )

        # The conversation has multiple violations, should score poorly
        assert result.overall_score < 70.0  # Below passing threshold
        assert len(result.compliance_flags) > 0  # Should have compliance violations
        # Verify specific compliance issues were detected
        assert ComplianceFlag.INAPPROPRIATE_LANGUAGE in result.compliance_flags
        assert ComplianceFlag.PII_EXPOSURE in result.compliance_flags

    # TODO: Add more QA scoring tests
    # - Test custom rule addition ?

    # - 1 Test individual rule evaluations
    def test_missing_greeting_reduces_greeting_score(self, qa_engine):
        """Conversation without greeting should score low in Greeting category"""

        speakers = [
            {"id": "agent_1", "role": "agent"},
            {"id": "customer_1", "role": "customer"},
        ]

        utterances = [
            {
                "speaker_id": "agent_1",
                "text": "What do you want?",
                "start_time": 0,
                "end_time": 2,
            },
            {
                "speaker_id": "customer_1",
                "text": "I need help with my order.",
                "start_time": 3,
                "end_time": 6,
            },
        ]

        result = qa_engine.score_conversation(
            conversation_id="no_greeting_test",
            speakers=speakers,
            utterances=utterances,
            metadata=None,
        )

        greeting_score = result.category_scores.get(ScoreCategory.GREETING)

        assert greeting_score is not None
        assert greeting_score <= 75

    def test_greeting_present_scores_full(self, qa_engine):
        """Test: Greeting rule passes when present """
        speakers = [
            {"id": "agent_1", "role": "agent"},
            {"id": "customer_1", "role": "customer"},
        ]

        utterances = [
            {
                "speaker_id": "agent_1",
                "text": "Hello, thank you for calling. My name is John.",
                "start_time": 0,
                "end_time": 2,
            },
            {
                "speaker_id": "customer_1",
                "text": "I need help.",
                "start_time": 3,
                "end_time": 5,
            },
        ]

        result = qa_engine.score_conversation(
            "greeting_test",
            speakers,
            utterances,
            None,
        )

        assert result.category_scores[ScoreCategory.GREETING] == 100.0

    def test_profanity_triggers_compliance_flag(self, qa_engine):
        """Test: Profanity triggers deduction + compliance flag"""
        speakers = [
            {"id": "agent_1", "role": "agent"},
            {"id": "customer_1", "role": "customer"},
        ]

        utterances = [
            {
                "speaker_id": "agent_1",
                "text": "This is stupid.",
                "start_time": 0,
                "end_time": 2,
            }
        ]

        result = qa_engine.score_conversation(
            "profanity_test",
            speakers,
            utterances,
            None,
        )

        assert ScoreCategory.PROFESSIONALISM in result.category_scores
        assert ComplianceFlag.INAPPROPRIATE_LANGUAGE in result.compliance_flags
        assert result.category_scores[ScoreCategory.PROFESSIONALISM] < 100

    # 2 Test category score breakdowns
    def test_category_score_never_negative(self, qa_engine):
        """Test category score cannot go below zero"""
        speakers = [
            {"id": "agent_1", "role": "agent"},
            {"id": "customer_1", "role": "customer"},
        ]

        # Intentionally violate multiple rules
        utterances = [
            {
                "speaker_id": "agent_1",
                "text": "damn hell crap stupid idiot 1234567890123456",
                "start_time": 0,
                "end_time": 2,
            }
        ]

        result = qa_engine.score_conversation(
            "extreme_violation_test",
            speakers,
            utterances,
            None,
        )

        for score in result.category_scores.values():
            assert score >= 0

    # 3 Test edge cases (empty conversations, etc.)
    def test_empty_conversation(self, qa_engine):
        result = qa_engine.score_conversation(
            "empty_test",
            speakers=[],
            utterances=[],
            metadata=None,
        )

        assert result.overall_score <= 100
        assert isinstance(result.category_scores, dict)


# =============================================================================
# Sample Security Tests - These are critical, expand significantly
# =============================================================================


class TestAPIAuthentication:
    """Tests for API authentication"""

    @pytest.mark.asyncio
    async def test_valid_api_key_authenticates(self):
        """Valid API key should authenticate successfully"""
        result = await validate_api_key("CUS_test_customer_123")

        assert result["customer_id"] == "cust_001"
        assert result["key_type"] == "customer"

    @pytest.mark.asyncio
    async def test_invalid_api_key_fails(self):
        """Invalid API key should raise 401"""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await validate_api_key("INVALID_KEY")

        assert exc_info.value.status_code == 401

    # TODO: Add more authentication tests
    # - Test suspended key handling
    # - Test admin vs customer key access
    # - Test tenant isolation

    # ============================================================
    # 1 Suspended Key Handling
    # ============================================================
    def test_suspended_key_rejected(self, client):
        """Suspended API key should return 403"""
        response = client.get(
            "/api/v1/conversations/",
            headers={"X-API-Key": "CUS_suspended_key"},
        )

        assert response.status_code == 403
        assert "not active" in response.json()["detail"]

    # ============================================================
    # 2️ Admin vs Customer Key Access
    # ============================================================

    def test_admin_cannot_access_customer_endpoint(self, client):
        """
        Admin key should not be allowed to access customer-only endpoints
        """
        response = client.get(
            "/api/v1/conversations/",
            headers={"X-API-Key": "ADM_admin_key_456"},
        )

        assert response.status_code == 403
        assert "requires a customer API key" in response.json()["detail"]

    def test_customer_cannot_access_admin_endpoint(self, client):
        """
        Customer key should not access admin-only endpoints
        """
        response = client.get(
            "/api/v1/conversations/admin/stats",
            headers={"X-API-Key": "CUS_test_customer_123"},
        )

        assert response.status_code == 403
        assert "requires an admin API key" in response.json()["detail"]

    def test_admin_can_access_admin_endpoint(self, client):
        """Admin key should access admin stats"""
        response = client.get(
            "/api/v1/conversations/admin/stats",
            headers={"X-API-Key": "ADM_admin_key_456"},
        )

        assert response.status_code == 200
        assert "total_conversations" in response.json()

    # ============================================================
    # 3️⃣ Tenant Isolation
    # ============================================================

    def test_customer_cannot_access_other_customer_conversation(self, client):
        """
        Customer should not access another tenant's conversation
        """
        # conv_001 belongs to cust_001
        response = client.get(
            "/api/v1/conversations/conv_001",
            headers={"X-API-Key": "CUS_premium_customer"},  # cust_002
        )

        # Should return 404 (not 403) to avoid information leakage
        assert response.status_code == 404

    def test_customer_can_access_own_conversation(self, client):
        """Customer should access their own conversation"""
        response = client.get(
            "/api/v1/conversations/conv_001",
            headers={"X-API-Key": "CUS_test_customer_123"},
        )

        assert response.status_code == 200
        assert response.json()["id"] == "conv_001"

    def test_list_conversations_is_tenant_isolated(self, client):
        """
        Customer listing should only return their conversations
        """
        response = client.get(
            "/api/v1/conversations/",
            headers={"X-API-Key": "CUS_test_customer_123"},
        )

        assert response.status_code == 200

        results = response.json()

        # All returned conversations must belong to cust_001
        for conv in results:
            assert conv["customer_id"] == "cust_001"

#=============================================================================
#SECURITY TESTS (BUG DETECTION)
#=============================================================================

class TestNotificationSecurityBugs:
    """Security tests targeting multi-tenant and mutation vulnerabilities"""

    def test_delete_rule_allows_cross_tenant_deletion(self, notification_service):
        """
        BUG-006:
        A user should NOT be able to delete another customer's rule.
        Current implementation ignores customer_id validation.
        """
        # Create rule for customer A
        rule = (
            RuleBuilder("cust_A")
            .with_name("Tenant Rule")
            .when_metric("qa_score")
            .is_less_than(50.0)
            .notify_via([NotificationChannel.EMAIL])
            .build()
        )
        notification_service.create_rule(rule)

        # Attempt delete using different customer
        deleted = notification_service.delete_rule(rule.id, customer_id="cust_B")

        # This should be False, but current implementation returns True
        assert deleted is False, "Cross-tenant deletion vulnerability detected"

    def test_update_rule_allows_customer_id_mutation(self, notification_service):
        """
        BUG-007:
        update_rule allows mutation of immutable fields like customer_id.
        """
        rule = (
            RuleBuilder("cust_001")
            .with_name("Immutable Test")
            .when_metric("qa_score")
            .is_less_than(60)
            .notify_via([NotificationChannel.EMAIL])
            .build()
        )
        notification_service.create_rule(rule)

        # Attempt malicious update
        updated = notification_service.update_rule(
            rule.id, {"customer_id": "hacked_customer"}
        )

        # This should not be allowed
        assert updated.customer_id == "cust_001", \
            "Rule ownership was mutated — security violation"


#=============================================================================
# UNIT TESTS (BUG DETECTION)
#=============================================================================
class TestNotificationEdgeCases:
    """Unit tests that expose edge-case bugs"""

    @pytest.mark.parametrize("invalid_channels", [None, []])
    def test_validate_rule_channels_null_safety(self, notification_service, invalid_channels):
        """
        BUG-008:
        Validation should handle None channels safely.
        """
        rule = NotificationRule(
            id="r1",
            name="Invalid Channels",
            customer_id="cust_001",
            enabled=True,
            metric="qa_score",
            operator=TriggerOperator.LESS_THAN,
            threshold=50,
            time_window_minutes=5,
            min_occurrences=1,
            channels=invalid_channels,
            severity=AlertSeverity.WARNING
        )

        with pytest.raises(ValueError):
            notification_service.create_rule(rule)


#=============================================================================
#INTEGRATION TESTS
#=============================================================================

class TestWebhookSerialization:
    """Integration-style tests for dispatch pipeline"""

    def test_webhook_fails_on_non_serializable_data(self, notification_service):
        """
        BUG-010:
        json.dumps should fail if event.data contains non-serializable objects.
        """

        class NonSerializable:
            pass

        event = notification_service._create_event(
            rule=(
                RuleBuilder("cust_001")
                .with_name("Serialization Test")
                .when_metric("qa_score")
                .is_less_than(60)
                .notify_via([NotificationChannel.WEBHOOK])
                .build()
            ),
            matching_data=[{"conversation_id": "conv_1"}]
        )

        # Inject non-serializable object
        event.data["bad"] = NonSerializable()

        with pytest.raises(TypeError):
            notification_service._send_webhook(event)


#=============================================================================
#API BUG TEST (BUG-004)
#=============================================================================

class TestSearchAPIBugs:
    """Integration tests targeting search filtering logic"""

    @pytest.mark.asyncio
    async def test_min_qa_score_zero_is_respected(self):
        """
        BUG-004:
        min_qa_score=0.0 should not be ignored due to truthiness check.
        """

        from src.api.conversation_api import search_conversations, SearchRequest

        request = SearchRequest(
            query="account",
            min_qa_score=0.0
        )

        results = await search_conversations(
            search_request=request,
            customer_id="cust_001"
        )

        print("Total results", results.total)
        assert results.total >= 1, \
            "min_qa_score=0.0 incorrectly ignored due to falsy check"


#=============================================================================
# QA ENGINE BUG TEST
#=============================================================================

class TestQANullSafety:
    """Tests exposing missing null-safety in _prepare_conversation_data"""

    def test_missing_text_key_raises_keyerror(self, qa_engine):
        """
        BUG-012:
        Missing 'text' key in utterance should not crash scoring.
        """

        speakers = [{"id": "a1", "role": "agent"}]
        utterances = [{"speaker_id": "a1"}]  # Missing text/start_time/end_time

        with pytest.raises(KeyError):
            qa_engine.score_conversation(
                conversation_id="conv_x",
                speakers=speakers,
                utterances=utterances
            )

# =============================================================================
# Placeholder for Performance Tests
# =============================================================================

class TestPerformance:
    """Performance test placeholders - implement using appropriate tooling"""

    @pytest.mark.skip(reason="Implement with locust or similar")
    def test_api_response_time_under_threshold(self):
        """API should respond within acceptable time"""
        # Use locust, k6, or similar for actual implementation
        pass

    @pytest.mark.skip(reason="Implement with profiling")
    def test_qa_scoring_performance(self):
        """QA scoring should complete within time budget"""
        pass


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
