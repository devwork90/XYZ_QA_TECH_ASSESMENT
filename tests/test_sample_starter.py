"""
XYZ Analytics - Sample Test Suite

This file contains starter tests to help you understand the testing patterns
and frameworks expected for this assessment.

Your task is to significantly expand upon these tests as outlined in the
assessment instructions.
"""

import pytest
import json
import os
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch, MagicMock

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
    # - Test time window aggregation
    # - Test cooldown period
    # - Test multiple rules evaluating same data


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
    # - Test individual rule evaluations
    # - Test category score breakdowns
    # - Test edge cases (empty conversations, etc.)
    # - Test custom rule addition


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
