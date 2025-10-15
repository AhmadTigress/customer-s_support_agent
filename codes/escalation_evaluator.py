import logging
import re
import numpy as np
from typing import Dict, Tuple, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class EscalationEvaluator:
    """Requires for human intervention based on chat context"""
    def __init__(self):
        self.conversation_history = {}


    def evaluate_escalation_need(self,
                                 current_message: str,
                                 conversation_history: List[Dict],
                                 sender: str,
                                 current_ai_response: str = "") -> Tuple[bool, str, float]:
        """
        Comprehensive evaluation for human escalation need.
        Return: (sholuld_escalate, reason, confidence_score)
        """
        escalation_factors = []

        # Factor 1: Conversation history analysis
        history_score, history_reason = self._analyse_conversation_history(conversation_history, sender)
        if history_score > 0.7:
            escalation_factors.append(history_score, history_reason)

        # Factor 2: User sentiment and emotion
        sentiment_score, sentiment_reason = self._analyse_sentiment(current_message, conversation_history)
        if sentiment_score > 0.8:
            escalation_factors.append((sentiment_score, sentiment_reason))

        # Factor 3: Complexity analysis
        complexity_score, complexity_reason = self._analyse_complexity(current_message, conversation_history)
        if complexity_score > 0.7:
            escalation_factors.append((complexity_score, complexity_reason))
        
        # Factor 4: Explicit Human Requests (with context awareness)
        explicit_score, explicit_reason = self._detect_explicit_human_requests(current_message, conversation_history)
        if explicit_score > 0.6:
            escalation_factors.append((explicit_score, explicit_reason))

        # Factor 6: AI Confidence and Capability
        capability_score, capability_reason = self._assess_ai_capability(current_message, current_ai_response)
        if capability_score > 0.7:
            escalation_factors.append((capability_score, capability_reason))

        # Calculate overall escalation score
        if escalation_factors:
            max_score, primary_reason = max(escalation_factors, key=lambda x: x[0])
            overall_score = self._calculate_composite_score(escalation_factors)
            
            should_escalate = overall_score >= 0.65  # Adjust threshold as needed
            
            return should_escalate, primary_reason, overall_score
        
        return False, "No significant escalation factors detected", 0.0
        
    

    def _analyse_conversation_history(self, conversation_history: List[Dict], sender: str) ->Tuple[float, str]:
        """Analyse conversation patterns that indicate escalation need"""
        if len(conversation_history) < 3:
            return 0.0, "Insufficient conversation history"
        
        # Track repeated questions/unsolved issues
        recent_messages = conversation_history[-6:]
        user_messages = [msg for msg in recent_messages if msg.get('type') == 'human']

        # Check for repetition
        unique_questions = set()
        repeated_issues = 0

        for msg in user_messages:
            content = msg.get('content', '').lower()
            content_hash = hash(content[:100])
            if content_hash in unique_questions:
                repeated_issues += 1
            else:
                unique_questions.add(content_hash)

        repetition_ratio = repeated_issues / len(user_messages) if user_messages else 0

        # Check for long conversation without resolution
        if len(conversation_history) > 10:
            return 0.8, f"Long conversation ({len(conversation_history)} messages) without resolution"
        
        if repetition_ratio > 0.3:
            return 0.3, f"User repeating issues (repetition rate: {repetition_ratio: 2f})"
        
        return 0.0, "Conversation history normal"
    

    def _analyse_sentiment(self, current_message: str, conversation_history: List[Dict]) -> Tuple[float, str]:
        """Analyse user sentiment and emotional state"""
        message_lower = current_message.lower()

        # Strong negative indicators
        strong_negative = [
                'angry', 'furious', 'livid', 'outraged', 'horrible', 'terrible',
                'awful', 'disgusting', 'ridiculous', 'unacceptable', 'worst ever',
                'never again', 'hate this', 'useless', 'waste of time'
        ]

        # Frustration indicators
        frustration_indicators = [
                'frustrated', 'annoyed', 'disappointed', 'not happy', 'not satisfied',
                'still not working', 'again', 'still having', 'why is this',
                'how many times', 'when will this be fixed'
        ]

        # Check for strong negative language
        strong_negative_count = sum(1 for word in strong_negative if word in message_lower)
        if strong_negative_count >= 2:
            return 0.9, "User expressing strong negative emotions"
    
        # Check for frustration patterns
        frustration_count = sum(1 for word in frustration_indicators if word in message_lower)
        if frustration_count >= 2:
            return 0.75, "User showing clear frustration"
    
        # Check for multiple exclamation points
        if current_message.count('!') >= 3:
            return 0.7, "User using excessive exclamation (emotional intensity)"
        
        return 0.0, "User sentiment appear neutral"
    

    def _analyse_complexity(self, current_message: str, conversation_history: List[Dict]) -> Tuple[float, str]:
        """Analyse query complexity that might require human expertise"""
        message_lower = current_message.lower()

        # High complexity topics
        high_complexity_indicators = {
            'legal': ['contract', 'agreement', 'terms', 'legal', 'liability', 'warranty', 'sue', 'lawyer'],
            'financial': ['refund', 'compensation', 'billing dispute', 'payment issue', 'chargeback', 'invoice'],
            'technical_advanced': ['api integration', 'custom development', 'system architecture', 'database', 'server'],
            'business_critical': ['downtime', 'outage', 'data loss', 'security breach', 'emergency'],
            'multi_step': ['process', 'workflow', 'multiple systems', 'integration between']
        }

        complexity_score = 0.0
        complexity_reasons = []

        for category, keywords in high_complexity_indicators.items():
            matches = sum(1 for keyword in keywords if keyword in message_lower)
            if matches > 0:
                category_score = min(0.3 + (matches * 0.2), 0.9)
                complexity_score = max(complexity_score, category_score)
                complexity_reasons.append(f"{category} issues deteted")

        # Check for multi-part questions
        if (' and ' in message_lower or ' also ' in message_lower) and message_lower.count('?') >= 2:
            complexity_score = max(complexity_score, 0.6)
            complexity_reasons.append("Multi-part complex questions")

        if complexity_score > 0.6:
            return complexity_score, f"Complex issues requiring expertise: {', '.join(complexity_reasons)}"
        
        return 0.0, "Query complexity within AI capabaility"


    def _detect_explicit_human_requests(self, current_message: str, conversation_history: List[Dict]) -> Tuple[float, str]:
        """Detect explicit human request with context awarenes"""
        message_lower = current_message.lower()

        # Direct human request
        direct_requests = [
            'speak to a human', 'talk to a real person', 'human agent', 
            'real person', 'live agent', 'customer service', 'support agent'
        ]

        # Check if this is a repeated request for human
        human_request_history = 0
        for msg in conversation_history[-4:]:
            if any(req in msg.get('content', '').lower() for req in direct_requests):
                human_request_history += 1

        # Current requests
        current_request = any(req in message_lower for req in direct_requests)

        if current_request and human_request_history >= 1:
            return 0.9, "Repeated explicit rquest for human assistance."
        elif current_request:
            return 0.7, "Explicit request for human assistance"
        
        # Indirect human requests
        indirect_requests = [
            'can you actually help', 'are you a bot', 'is this automated',
            'let me speak to someone', 'get me a manager', 'supervisor'
        ]

        if any(req in message_lower for req in indirect_requets):
            return 0.6, "Indirec trequest for human assistance"
        
        return 0.0, "No explicit request for human assistance"
    

    def _assess_ai_capability(self, current_message: str, current_ai_response: str) -> Tuple[float, str]:
        """Assess whether this query is within AI capability"""
        message_lower = current_message.lower()

        # Queries beyond typical AI capabiltiies
        beyond_ai_capability = [
            'make an exception', 'override', 'special case', 'discretion',
            'judgment call', 'subjective', 'personal opinion', 'what would you do',
            'emotional support', 'counseling', 'therapy'
        ]

        if any(phrase in message_lower for phrase in beyond_ai_capability):
            return 0.8, "Query requires human judgement and discretion"
        
        # Check if AI response require/indicate uncertainity
        if current_ai_response:
            uncertainity_indicators =  [
                "I'm not sure", "I don't know", "I cannot", "unable to",
                "limited information", "contact support", "escalate"
            ]

            if any(indicator in current_ai_response.lower() for indicator in uncertainity_indicators):
                return 0.7, "AI response indicates uncertainity or limitations"
            
            return 0.0, "Query appears within AI capabilities"


    def _calculate_composite_score(self, factors: List[Tuple[float, str]]) -> float:
        """Calculate weighted composite escalation score"""
        if not factors:
            return 0.0
        
        # Use maximum score with slight weighting toward multiple factors
        max_score = max(score for score, _ in factors)
        factor_count_bonus = min(len(factors) * 0.1, 0.3)

        return min(max_score + factor_count_bonus, 1.0)