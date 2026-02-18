"""
XYZ Analytics - Automated QA Scoring Engine

This module implements the Automated Quality Assurance (AQA) scoring system.
It evaluates conversation quality across multiple dimensions using both
rule-based checks and ML-powered analysis.

Note: This is a simplified version for assessment purposes.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
import re


class ScoreCategory(Enum):
    """Categories for QA scoring breakdown"""
    GREETING = "greeting"
    PROFESSIONALISM = "professionalism"
    RESOLUTION = "resolution"
    COMPLIANCE = "compliance"
    EMPATHY = "empathy"
    CLARITY = "clarity"


class ComplianceFlag(Enum):
    """Compliance violation types"""
    PII_EXPOSURE = "pii_exposure"
    UNAUTHORIZED_PROMISE = "unauthorized_promise"
    INAPPROPRIATE_LANGUAGE = "inappropriate_language"
    MISSING_DISCLOSURE = "missing_disclosure"
    HOLD_TIME_VIOLATION = "hold_time_violation"


@dataclass
class QAResult:
    """
    Complete QA scoring result for a conversation.
    """
    conversation_id: str
    overall_score: float  # 0-100
    category_scores: Dict[ScoreCategory, float]
    compliance_flags: List[ComplianceFlag]
    deductions: List[Dict[str, Any]]
    recommendations: List[str]
    processed_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ScoringRule:
    """
    A rule that evaluates a specific aspect of conversation quality.
    """
    id: str
    name: str
    category: ScoreCategory
    weight: float  # How much this affects the category score (0-1)
    check_type: str  # "regex", "keyword", "sentiment", "timing", "custom"
    parameters: Dict[str, Any]
    deduction_points: float  # Points to deduct if rule fails
    is_compliance_rule: bool = False
    compliance_flag: Optional[ComplianceFlag] = None


class QAScoringEngine:
    """
    Main scoring engine that orchestrates QA evaluation.
    
    The engine:
    1. Applies configurable scoring rules
    2. Calculates category-specific scores
    3. Aggregates to an overall score
    4. Generates recommendations
    """
    
    DEFAULT_CATEGORY_WEIGHTS = {
        ScoreCategory.GREETING: 0.10,
        ScoreCategory.PROFESSIONALISM: 0.25,
        ScoreCategory.RESOLUTION: 0.30,
        ScoreCategory.COMPLIANCE: 0.20,
        ScoreCategory.EMPATHY: 0.10,
        ScoreCategory.CLARITY: 0.05
    }
    
    def __init__(self):
        self.rules: List[ScoringRule] = []
        self.category_weights = self.DEFAULT_CATEGORY_WEIGHTS.copy()
        self._load_default_rules()
    
    def _load_default_rules(self):
        """Load the default set of QA scoring rules"""
        
        # Greeting rules
        self.rules.extend([
            ScoringRule(
                id="greeting_present",
                name="Opening Greeting Present",
                category=ScoreCategory.GREETING,
                weight=0.6,
                check_type="regex",
                parameters={
                    "pattern": r"(?i)(hello|hi|good\s*(morning|afternoon|evening)|thank\s*you\s*for\s*(calling|contacting))",
                    "target": "agent_first_utterance"
                },
                deduction_points=15
            ),
            ScoringRule(
                id="agent_name_given",
                name="Agent Identifies Themselves",
                category=ScoreCategory.GREETING,
                weight=0.4,
                check_type="regex",
                parameters={
                    "pattern": r"(?i)(my\s*name\s*is|this\s*is|i\'m|speaking\s*with)",
                    "target": "agent_first_utterance"
                },
                deduction_points=10
            ),
        ])
        
        # Professionalism rules
        self.rules.extend([
            ScoringRule(
                id="no_profanity",
                name="No Inappropriate Language",
                category=ScoreCategory.PROFESSIONALISM,
                weight=0.5,
                check_type="keyword",
                parameters={
                    "forbidden_words": ["damn", "hell", "crap", "stupid", "idiot"],
                    "target": "agent_utterances"
                },
                deduction_points=25,
                is_compliance_rule=True,
                compliance_flag=ComplianceFlag.INAPPROPRIATE_LANGUAGE
            ),
            ScoringRule(
                id="professional_tone",
                name="Maintains Professional Tone",
                category=ScoreCategory.PROFESSIONALISM,
                weight=0.3,
                check_type="sentiment",
                parameters={
                    "min_professionalism_score": 0.6,
                    "target": "agent_utterances"
                },
                deduction_points=15
            ),
            ScoringRule(
                id="no_interruptions",
                name="Does Not Interrupt Customer",
                category=ScoreCategory.PROFESSIONALISM,
                weight=0.2,
                check_type="timing",
                parameters={
                    "max_overlap_ratio": 0.1,
                    "min_gap_seconds": 0.3
                },
                deduction_points=10
            ),
        ])
        
        # Resolution rules
        self.rules.extend([
            ScoringRule(
                id="issue_acknowledged",
                name="Issue Acknowledged",
                category=ScoreCategory.RESOLUTION,
                weight=0.3,
                check_type="regex",
                parameters={
                    "pattern": r"(?i)(understand|see|got\s*it|i\s*hear\s*you|that\s*makes\s*sense)",
                    "target": "agent_utterances"
                },
                deduction_points=10
            ),
            ScoringRule(
                id="solution_provided",
                name="Solution or Next Steps Provided",
                category=ScoreCategory.RESOLUTION,
                weight=0.5,
                check_type="regex",
                parameters={
                    "pattern": r"(?i)(here\'s\s*what|let\s*me|i\s*(can|will)|next\s*step|solution|resolve)",
                    "target": "agent_utterances"
                },
                deduction_points=20
            ),
            ScoringRule(
                id="closing_summary",
                name="Closing Summary Given",
                category=ScoreCategory.RESOLUTION,
                weight=0.2,
                check_type="regex",
                parameters={
                    "pattern": r"(?i)(to\s*summarize|in\s*summary|we\'ve\s*(discussed|covered)|anything\s*else)",
                    "target": "agent_last_utterances"
                },
                deduction_points=5
            ),
        ])
        
        # Compliance rules
        self.rules.extend([
            ScoringRule(
                id="no_pii_exposure",
                name="No PII Exposure",
                category=ScoreCategory.COMPLIANCE,
                weight=0.4,
                check_type="regex",
                parameters={
                    "patterns": [
                        r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b",  # SSN
                        r"\b\d{16}\b",  # Credit card
                        r"\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b",  # Credit card with spaces
                    ],
                    "target": "agent_utterances"
                },
                deduction_points=50,
                is_compliance_rule=True,
                compliance_flag=ComplianceFlag.PII_EXPOSURE
            ),
            ScoringRule(
                id="no_unauthorized_promises",
                name="No Unauthorized Commitments",
                category=ScoreCategory.COMPLIANCE,
                weight=0.3,
                check_type="keyword",
                parameters={
                    "forbidden_phrases": [
                        "i guarantee",
                        "i promise",
                        "definitely will",
                        "100% certain",
                        "money back guarantee"
                    ],
                    "target": "agent_utterances"
                },
                deduction_points=30,
                is_compliance_rule=True,
                compliance_flag=ComplianceFlag.UNAUTHORIZED_PROMISE
            ),
            ScoringRule(
                id="hold_time_compliance",
                name="Hold Time Under Limit",
                category=ScoreCategory.COMPLIANCE,
                weight=0.3,
                check_type="timing",
                parameters={
                    "max_hold_seconds": 180,
                    "target": "hold_periods"
                },
                deduction_points=20,
                is_compliance_rule=True,
                compliance_flag=ComplianceFlag.HOLD_TIME_VIOLATION
            ),
        ])
        
        # Empathy rules
        self.rules.extend([
            ScoringRule(
                id="empathy_shown",
                name="Empathy Statements Present",
                category=ScoreCategory.EMPATHY,
                weight=0.6,
                check_type="regex",
                parameters={
                    "pattern": r"(?i)(sorry|understand|frustrat|difficult|appreciate|thank\s*you\s*for\s*your\s*patience)",
                    "target": "agent_utterances"
                },
                deduction_points=15
            ),
            ScoringRule(
                id="active_listening",
                name="Active Listening Demonstrated",
                category=ScoreCategory.EMPATHY,
                weight=0.4,
                check_type="custom",
                parameters={
                    "check_function": "check_active_listening"
                },
                deduction_points=10
            ),
        ])
        
        # Clarity rules
        self.rules.extend([
            ScoringRule(
                id="clear_language",
                name="Uses Clear Language",
                category=ScoreCategory.CLARITY,
                weight=0.5,
                check_type="custom",
                parameters={
                    "max_avg_sentence_length": 25,
                    "target": "agent_utterances"
                },
                deduction_points=10
            ),
            ScoringRule(
                id="no_jargon",
                name="Avoids Excessive Jargon",
                category=ScoreCategory.CLARITY,
                weight=0.5,
                check_type="keyword",
                parameters={
                    "jargon_threshold": 3,
                    "jargon_words": ["leverage", "synergy", "bandwidth", "circle back", "ping"],
                    "target": "agent_utterances"
                },
                deduction_points=5
            ),
        ])
    
    def score_conversation(
        self, 
        conversation_id: str,
        speakers: List[Dict],
        utterances: List[Dict],
        metadata: Optional[Dict] = None
    ) -> QAResult:
        """
        Score a conversation against all QA rules.
        
        This is the main entry point for QA scoring.
        
        Args:
            conversation_id: Unique identifier for the conversation
            speakers: List of speaker objects with id and role
            utterances: List of utterance objects with speaker_id, text, start_time, end_time
            metadata: Optional additional metadata
            
        Returns:
            QAResult with scores, flags, and recommendations
        """
        # Prepare conversation data for analysis
        conversation_data = self._prepare_conversation_data(speakers, utterances)
        
        # Initialize scoring
        category_scores = {cat: 100.0 for cat in ScoreCategory}
        deductions = []
        compliance_flags = []
        
        # Evaluate each rule
        for rule in self.rules:
            passed, details = self._evaluate_rule(rule, conversation_data)
            
            if not passed:
                # Apply deduction to category
                category_scores[rule.category] -= rule.deduction_points
                
                deductions.append({
                    "rule_id": rule.id,
                    "rule_name": rule.name,
                    "category": rule.category.value,
                    "points_deducted": rule.deduction_points,
                    "details": details
                })
                
                # Track compliance violations
                if rule.is_compliance_rule and rule.compliance_flag:
                    compliance_flags.append(rule.compliance_flag)
        
        # Ensure category scores don't go below 0
        for cat in category_scores:
            category_scores[cat] = max(0, category_scores[cat])
        
        # Calculate weighted overall score
        overall_score = self._calculate_overall_score(category_scores)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(deductions, category_scores)
        
        return QAResult(
            conversation_id=conversation_id,
            overall_score=overall_score,
            category_scores=category_scores,
            compliance_flags=compliance_flags,
            deductions=deductions,
            recommendations=recommendations
        )
    
    def _prepare_conversation_data(
        self, 
        speakers: List[Dict], 
        utterances: List[Dict]
    ) -> Dict[str, Any]:
        """
        Prepare conversation data into a format suitable for rule evaluation.
        
        BUG: There's an issue in this method. Can you identify it?
        """
        # Identify agent and customer speakers
        agent_ids = [s["id"] for s in speakers if s["role"] == "agent"]
        customer_ids = [s["id"] for s in speakers if s["role"] == "customer"]
        
        # Separate utterances by role
        agent_utterances = [u for u in utterances if u["speaker_id"] in agent_ids]
        customer_utterances = [u for u in utterances if u["speaker_id"] in customer_ids]
        
        # Get first and last agent utterances
        agent_first = agent_utterances[0] if agent_utterances else None
        agent_last_utterances = agent_utterances[-3:] if agent_utterances else []
        
        # Concatenate all agent text
        agent_text = " ".join([u["text"] for u in agent_utterances])
        customer_text = " ".join([u["text"] for u in customer_utterances])
        
        # Calculate timing metrics
        hold_periods = self._detect_hold_periods(utterances)
        overlap_ratio = self._calculate_overlap_ratio(utterances)
        
        return {
            "speakers": speakers,
            "utterances": utterances,
            "agent_utterances": agent_utterances,
            "customer_utterances": customer_utterances,
            "agent_first_utterance": agent_first,
            "agent_last_utterances": agent_last_utterances,
            "agent_text": agent_text,
            "customer_text": customer_text,
            "hold_periods": hold_periods,
            "overlap_ratio": overlap_ratio
        }
    
    def _evaluate_rule(
        self, 
        rule: ScoringRule, 
        data: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Evaluate a single rule against the conversation data.
        
        Returns (passed: bool, details: str)
        """
        check_type = rule.check_type
        params = rule.parameters
        
        if check_type == "regex":
            return self._check_regex(rule, data)
        elif check_type == "keyword":
            return self._check_keywords(rule, data)
        elif check_type == "timing":
            return self._check_timing(rule, data)
        elif check_type == "sentiment":
            return self._check_sentiment(rule, data)
        elif check_type == "custom":
            return self._check_custom(rule, data)
        else:
            return True, "Unknown check type"
    
    def _check_regex(self, rule: ScoringRule, data: Dict) -> Tuple[bool, str]:
        """Check if a regex pattern matches the target text"""
        params = rule.parameters
        target = params.get("target", "agent_text")
        
        # Get the target text
        if target == "agent_first_utterance":
            text_data = data.get("agent_first_utterance")
            if not text_data:
                return False, "No agent utterance found"
            text = text_data.get("text", "")
        elif target == "agent_last_utterances":
            utterances = data.get("agent_last_utterances", [])
            text = " ".join([u.get("text", "") for u in utterances])
        elif target == "agent_utterances":
            text = data.get("agent_text", "")
        else:
            text = data.get(target, "")
        
        # Handle multiple patterns (for compliance)
        if "patterns" in params:
            for pattern in params["patterns"]:
                if re.search(pattern, text):
                    return False, f"Pattern matched: {pattern}"
            return True, "No forbidden patterns found"
        
        # Single pattern check
        pattern = params.get("pattern", "")
        if re.search(pattern, text):
            return True, "Pattern matched"
        return False, f"Pattern not found: {pattern[:50]}..."
    
    def _check_keywords(self, rule: ScoringRule, data: Dict) -> Tuple[bool, str]:
        """Check for presence/absence of keywords"""
        params = rule.parameters
        target = params.get("target", "agent_text")
        
        if target == "agent_utterances":
            text = data.get("agent_text", "").lower()
        else:
            text = data.get(target, "").lower()
        
        # Forbidden words check
        if "forbidden_words" in params:
            for word in params["forbidden_words"]:
                if word.lower() in text:
                    return False, f"Forbidden word found: {word}"
            return True, "No forbidden words found"
        
        # Forbidden phrases check
        if "forbidden_phrases" in params:
            for phrase in params["forbidden_phrases"]:
                if phrase.lower() in text:
                    return False, f"Forbidden phrase found: {phrase}"
            return True, "No forbidden phrases found"
        
        # Jargon check
        if "jargon_words" in params:
            jargon_count = sum(1 for word in params["jargon_words"] if word.lower() in text)
            threshold = params.get("jargon_threshold", 3)
            if jargon_count >= threshold:
                return False, f"Excessive jargon: {jargon_count} instances"
            return True, f"Jargon within limits: {jargon_count}"
        
        return True, "No keyword issues"
    
    def _check_timing(self, rule: ScoringRule, data: Dict) -> Tuple[bool, str]:
        """Check timing-related metrics"""
        params = rule.parameters
        
        # Hold time check
        if "max_hold_seconds" in params:
            hold_periods = data.get("hold_periods", [])
            max_hold = params["max_hold_seconds"]
            
            for period in hold_periods:
                if period["duration"] > max_hold:
                    return False, f"Hold time exceeded: {period['duration']}s > {max_hold}s"
            return True, "Hold time within limits"
        
        # Overlap check
        if "max_overlap_ratio" in params:
            overlap = data.get("overlap_ratio", 0)
            max_overlap = params["max_overlap_ratio"]
            if overlap > max_overlap:
                return False, f"Overlap ratio too high: {overlap:.2f} > {max_overlap}"
            return True, f"Overlap ratio acceptable: {overlap:.2f}"
        
        return True, "Timing check passed"
    
    def _check_sentiment(self, rule: ScoringRule, data: Dict) -> Tuple[bool, str]:
        """
        Check sentiment-related metrics.
        
        Note: In production, this would use ML models. 
        For assessment purposes, we use a simplified heuristic.
        """
        # Simplified professionalism check based on positive phrases
        agent_text = data.get("agent_text", "").lower()
        
        professional_phrases = [
            "happy to help", "my pleasure", "certainly", "absolutely",
            "i'd be glad", "let me assist", "thank you for"
        ]
        
        score = sum(1 for phrase in professional_phrases if phrase in agent_text)
        normalized_score = min(1.0, score / 3)  # Cap at 1.0
        
        min_score = rule.parameters.get("min_professionalism_score", 0.6)
        
        if normalized_score >= min_score:
            return True, f"Professionalism score: {normalized_score:.2f}"
        return False, f"Low professionalism score: {normalized_score:.2f} < {min_score}"
    
    def _check_custom(self, rule: ScoringRule, data: Dict) -> Tuple[bool, str]:
        """Run custom check functions"""
        check_function = rule.parameters.get("check_function", "")
        
        if check_function == "check_active_listening":
            return self._check_active_listening(data)
        
        # Sentence length check
        if "max_avg_sentence_length" in rule.parameters:
            agent_text = data.get("agent_text", "")
            sentences = re.split(r'[.!?]+', agent_text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                return True, "No sentences to evaluate"
            
            avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
            max_length = rule.parameters["max_avg_sentence_length"]
            
            if avg_length <= max_length:
                return True, f"Average sentence length: {avg_length:.1f} words"
            return False, f"Sentences too long: {avg_length:.1f} > {max_length}"
        
        return True, "Custom check passed"
    
    def _check_active_listening(self, data: Dict) -> Tuple[bool, str]:
        """
        Check for active listening indicators.
        
        Looks for:
        - Paraphrasing customer statements
        - Clarifying questions
        - Acknowledgment phrases
        """
        agent_utterances = data.get("agent_utterances", [])
        
        indicators_found = 0
        
        listening_patterns = [
            r"(?i)so\s*(you\'re|you\s*are)\s*saying",
            r"(?i)let\s*me\s*(make\s*sure|confirm)",
            r"(?i)if\s*i\s*understand\s*(correctly|right)",
            r"(?i)can\s*you\s*(tell|explain|clarify)",
            r"(?i)i\s*hear\s*(you|that)"
        ]
        
        for utt in agent_utterances:
            text = utt.get("text", "")
            for pattern in listening_patterns:
                if re.search(pattern, text):
                    indicators_found += 1
                    break  # Count once per utterance
        
        # Expect at least 1 indicator for conversations with multiple turns
        min_indicators = max(1, len(agent_utterances) // 5)
        
        if indicators_found >= min_indicators:
            return True, f"Active listening indicators found: {indicators_found}"
        return False, f"Insufficient active listening: {indicators_found} < {min_indicators}"
    
    def _detect_hold_periods(self, utterances: List[Dict]) -> List[Dict]:
        """
        Detect periods where there was silence (potential hold).
        
        BUG: This implementation has a logical error. Can you spot it?
        """
        hold_periods = []
        sorted_utterances = sorted(utterances, key=lambda x: x["start_time"])
        
        for i in range(1, len(sorted_utterances)):
            prev_end = sorted_utterances[i-1]["end_time"]
            curr_start = sorted_utterances[i]["start_time"]
            
            gap = curr_start - prev_end
            
            # Consider gaps > 10 seconds as potential holds
            if gap > 10:
                hold_periods.append({
                    "start": prev_end,
                    "end": curr_start,
                    "duration": gap
                })
        
        return hold_periods
    
    def _calculate_overlap_ratio(self, utterances: List[Dict]) -> float:
        """Calculate the ratio of overlapping speech"""
        if len(utterances) < 2:
            return 0.0
        
        sorted_utterances = sorted(utterances, key=lambda x: x["start_time"])
        overlap_time = 0.0
        total_time = 0.0
        
        for i, utt in enumerate(sorted_utterances):
            duration = utt["end_time"] - utt["start_time"]
            total_time += duration
            
            # Check overlap with next utterance
            if i < len(sorted_utterances) - 1:
                next_utt = sorted_utterances[i + 1]
                if utt["end_time"] > next_utt["start_time"]:
                    overlap = min(utt["end_time"], next_utt["end_time"]) - next_utt["start_time"]
                    overlap_time += max(0, overlap)
        
        return overlap_time / total_time if total_time > 0 else 0.0
    
    def _calculate_overall_score(self, category_scores: Dict[ScoreCategory, float]) -> float:
        """
        Calculate weighted overall score from category scores.
        
        BUG: There's an issue with how the weights are applied. Can you find it?
        """
        total_score = 0.0
        total_weight = 0.0
        
        for category, score in category_scores.items():
            weight = self.category_weights.get(category, 0)
            total_score += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return round(total_score / total_weight, 2)
    
    def _generate_recommendations(
        self, 
        deductions: List[Dict], 
        category_scores: Dict[ScoreCategory, float]
    ) -> List[str]:
        """Generate actionable recommendations based on scoring results"""
        recommendations = []
        
        # Find lowest scoring categories
        sorted_categories = sorted(
            category_scores.items(), 
            key=lambda x: x[1]
        )
        
        for category, score in sorted_categories[:2]:  # Top 2 worst
            if score < 70:
                recommendations.append(
                    f"Focus on improving {category.value}: current score is {score:.1f}%"
                )
        
        # Specific recommendations based on deductions
        for deduction in deductions:
            rule_id = deduction["rule_id"]
            
            if rule_id == "greeting_present":
                recommendations.append(
                    "Always start with a proper greeting: 'Thank you for calling [Company], my name is [Name]'"
                )
            elif rule_id == "empathy_shown":
                recommendations.append(
                    "Use empathy statements like 'I understand how frustrating that must be'"
                )
            elif rule_id == "solution_provided":
                recommendations.append(
                    "Clearly state the solution or next steps before ending the call"
                )
            elif rule_id in ["no_pii_exposure", "no_unauthorized_promises"]:
                recommendations.append(
                    f"COMPLIANCE: Review training on {deduction['rule_name']}"
                )
        
        return list(set(recommendations))[:5]  # Dedupe and limit to 5
    
    def add_custom_rule(self, rule: ScoringRule) -> None:
        """Add a custom scoring rule"""
        self.rules.append(rule)
    
    def set_category_weight(self, category: ScoreCategory, weight: float) -> None:
        """
        Update the weight for a scoring category.
        
        BUG: Missing validation - what should be checked?
        """
        self.category_weights[category] = weight
    
    def get_rules_by_category(self, category: ScoreCategory) -> List[ScoringRule]:
        """Get all rules for a specific category"""
        return [r for r in self.rules if r.category == category]
