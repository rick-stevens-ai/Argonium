#!/usr/bin/env python3
"""
Enhanced Formal Logic Analyzer

This analyzer uses sophisticated pattern matching and natural language processing
techniques to identify formal logical argument structures without requiring
external dependencies like spaCy or NetworkX.

It focuses on identifying:
1. Classical logical argument forms (modus ponens, modus tollens, etc.)
2. Propositional logic structures
3. Quantified statements and their relationships
4. Inference patterns and logical validity
5. Argument strength and confidence measures
"""

import re
import json
import os
from typing import Dict, List, Tuple, Set, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
from enum import Enum

class ArgumentType(Enum):
    """Types of logical arguments"""
    MODUS_PONENS = "modus_ponens"
    MODUS_TOLLENS = "modus_tollens"
    HYPOTHETICAL_SYLLOGISM = "hypothetical_syllogism"
    DISJUNCTIVE_SYLLOGISM = "disjunctive_syllogism"
    CONTRAPOSITION = "contraposition"
    REDUCTIO_AD_ABSURDUM = "reductio_ad_absurdum"
    UNIVERSAL_INSTANTIATION = "universal_instantiation"
    EXISTENTIAL_GENERALIZATION = "existential_generalization"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    INDUCTIVE_GENERALIZATION = "inductive_generalization"
    ELIMINATION = "elimination"
    CONFIRMATION = "confirmation"
    COUNTERFACTUAL = "counterfactual"

@dataclass
class LogicalClause:
    """Represents a logical clause or proposition"""
    id: str
    text: str
    normalized_form: str
    predicate: str
    subjects: List[str]
    objects: List[str]
    modifiers: List[str]
    polarity: bool  # True for positive, False for negative
    certainty: float  # 0.0 to 1.0
    temporal_marker: Optional[str]
    conditional_type: Optional[str]  # 'if_then', 'unless', 'when', etc.
    quantifier: Optional[str]  # 'all', 'some', 'no', 'most', etc.
    sentence_position: int
    
    def to_dict(self):
        return asdict(self)

@dataclass
class LogicalArgument:
    """Represents a detected logical argument"""
    id: str
    argument_type: ArgumentType
    premises: List[LogicalClause]
    conclusion: LogicalClause
    intermediate_steps: List[LogicalClause]
    logical_form: str
    natural_language_form: str
    validity: bool
    soundness_estimate: float
    confidence: float
    text_span: Tuple[int, int]  # Start and end positions in original text
    inference_rules_used: List[str]

class LogicalPatternExtractor:
    """Extracts logical patterns using advanced regex and linguistic rules"""
    
    def __init__(self):
        self.setup_patterns()
    
    def setup_patterns(self):
        """Initialize pattern matching rules"""
        # Conditional patterns
        self.conditional_patterns = [
            r'if\s+(.+?)\s*,?\s*then\s+(.+?)[\.,]',
            r'if\s+(.+?)\s*,\s*(.+?)[\.,]',
            r'when\s+(.+?)\s*,\s*(.+?)[\.,]',
            r'whenever\s+(.+?)\s*,\s*(.+?)[\.,]',
            r'provided\s+that\s+(.+?)\s*,\s*(.+?)[\.,]',
            r'given\s+that\s+(.+?)\s*,\s*(.+?)[\.,]',
            r'assuming\s+(.+?)\s*,\s*(.+?)[\.,]'
        ]
        
        # Causal patterns
        self.causal_patterns = [
            r'(.+?)\s+causes?\s+(.+?)[\.,]',
            r'(.+?)\s+leads?\s+to\s+(.+?)[\.,]',
            r'(.+?)\s+results?\s+in\s+(.+?)[\.,]',
            r'because\s+(.+?)\s*,\s*(.+?)[\.,]',
            r'since\s+(.+?)\s*,\s*(.+?)[\.,]',
            r'due\s+to\s+(.+?)\s*,\s*(.+?)[\.,]'
        ]
        
        # Disjunctive patterns
        self.disjunctive_patterns = [
            r'either\s+(.+?)\s+or\s+(.+?)[\.,]',
            r'(.+?)\s+or\s+(.+?)[\.,]',
            r'alternatively\s*,\s*(.+?)[\.,]'
        ]
        
        # Negation patterns
        self.negation_patterns = [
            r'\b(?:not|no|never|nothing|nobody|nowhere|neither)\b',
            r'\b(?:isn\'t|aren\'t|wasn\'t|weren\'t|won\'t|wouldn\'t|can\'t|couldn\'t|shouldn\'t|mustn\'t)\b',
            r'\b(?:don\'t|doesn\'t|didn\'t|haven\'t|hasn\'t|hadn\'t)\b'
        ]
        
        # Certainty markers
        self.certainty_markers = {
            'high': ['definitely', 'certainly', 'clearly', 'obviously', 'undoubtedly', 'must', 'will', 'always'],
            'medium': ['probably', 'likely', 'should', 'would', 'generally', 'typically', 'usually'],
            'low': ['possibly', 'might', 'could', 'maybe', 'perhaps', 'sometimes', 'occasionally']
        }
        
        # Quantifier patterns
        self.quantifier_patterns = {
            'universal': ['all', 'every', 'each', 'any', 'always'],
            'existential': ['some', 'there is', 'there are', 'exists', 'sometimes'],
            'negative': ['no', 'none', 'nothing', 'never', 'nobody'],
            'proportional': ['most', 'many', 'few', 'several', 'majority']
        }
        
        # Inference indicators
        self.inference_indicators = [
            'therefore', 'thus', 'hence', 'consequently', 'so', 'it follows that',
            'we can conclude', 'this means', 'this shows', 'this indicates',
            'as a result', 'accordingly', 'for this reason'
        ]
        
        # Evidence patterns
        self.evidence_patterns = [
            r'the evidence shows that (.+?)[\.,]',
            r'research indicates that (.+?)[\.,]',
            r'studies show that (.+?)[\.,]',
            r'data suggests that (.+?)[\.,]',
            r'it has been demonstrated that (.+?)[\.,]'
        ]
        
        # Contradiction patterns
        self.contradiction_patterns = [
            r'but (.+?) contradicts (.+?)[\.,]',
            r'however\s*,\s*(.+?)[\.,]',
            r'on the other hand\s*,\s*(.+?)[\.,]',
            r'in contrast\s*,\s*(.+?)[\.,]',
            r'this contradicts (.+?)[\.,]'
        ]
    
    def extract_logical_clauses(self, text: str) -> List[LogicalClause]:
        """Extract logical clauses from text"""
        clauses = []
        sentences = self._split_into_sentences(text)
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short fragments
                continue
            
            # Extract basic clause information
            clause = self._analyze_sentence(sentence, i)
            if clause:
                clauses.append(clause)
                
            # Look for complex structures within sentences
            sub_clauses = self._extract_sub_clauses(sentence, i)
            clauses.extend(sub_clauses)
        
        return clauses
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using sophisticated rules"""
        # Handle common abbreviations and edge cases
        text = re.sub(r'([A-Z][a-z]{1,2}\.)', r'\\1', text)  # Dr. -> Dr\\.
        text = re.sub(r'(\w\.g\.|i\.e\.|etc\.)', r'\\1', text)  # e.g. -> e\\.g\\.
        
        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+(?:\s+|$)', text)
        
        # Restore abbreviations
        sentences = [s.replace('\\.', '.') for s in sentences if s.strip()]
        
        return sentences
    
    def _analyze_sentence(self, sentence: str, position: int) -> Optional[LogicalClause]:
        """Analyze a single sentence to extract logical structure"""
        # Normalize sentence
        normalized = sentence.lower().strip()
        
        # Extract components
        predicate = self._extract_main_predicate(sentence)
        subjects = self._extract_subjects(sentence)
        objects = self._extract_objects(sentence)
        modifiers = self._extract_modifiers(sentence)
        
        # Determine polarity
        polarity = not self._has_negation(sentence)
        
        # Estimate certainty
        certainty = self._estimate_certainty(sentence)
        
        # Detect temporal markers
        temporal_marker = self._extract_temporal_marker(sentence)
        
        # Detect conditional type
        conditional_type = self._detect_conditional_type(sentence)
        
        # Detect quantifiers
        quantifier = self._detect_quantifier(sentence)
        
        return LogicalClause(
            id=f"C{position}",
            text=sentence,
            normalized_form=normalized,
            predicate=predicate,
            subjects=subjects,
            objects=objects,
            modifiers=modifiers,
            polarity=polarity,
            certainty=certainty,
            temporal_marker=temporal_marker,
            conditional_type=conditional_type,
            quantifier=quantifier,
            sentence_position=position
        )
    
    def _extract_sub_clauses(self, sentence: str, position: int) -> List[LogicalClause]:
        """Extract sub-clauses from complex sentences"""
        sub_clauses = []
        
        # Look for conditional statements
        for pattern in self.conditional_patterns:
            matches = re.finditer(pattern, sentence, re.IGNORECASE)
            for match in matches:
                antecedent = match.group(1).strip()
                consequent = match.group(2).strip()
                
                # Create sub-clauses for antecedent and consequent
                if len(antecedent) > 5:
                    ant_clause = self._create_sub_clause(antecedent, f"{position}_if", "conditional_antecedent")
                    sub_clauses.append(ant_clause)
                
                if len(consequent) > 5:
                    cons_clause = self._create_sub_clause(consequent, f"{position}_then", "conditional_consequent")
                    sub_clauses.append(cons_clause)
        
        # Look for causal relationships
        for pattern in self.causal_patterns:
            matches = re.finditer(pattern, sentence, re.IGNORECASE)
            for match in matches:
                cause = match.group(1).strip()
                effect = match.group(2).strip()
                
                if len(cause) > 5:
                    cause_clause = self._create_sub_clause(cause, f"{position}_cause", "causal_antecedent")
                    sub_clauses.append(cause_clause)
                
                if len(effect) > 5:
                    effect_clause = self._create_sub_clause(effect, f"{position}_effect", "causal_consequent")
                    sub_clauses.append(effect_clause)
        
        # Look for disjunctive statements
        for pattern in self.disjunctive_patterns:
            matches = re.finditer(pattern, sentence, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 2:
                    option1 = match.group(1).strip()
                    option2 = match.group(2).strip()
                    
                    if len(option1) > 5:
                        opt1_clause = self._create_sub_clause(option1, f"{position}_or1", "disjunctive_option")
                        sub_clauses.append(opt1_clause)
                    
                    if len(option2) > 5:
                        opt2_clause = self._create_sub_clause(option2, f"{position}_or2", "disjunctive_option")
                        sub_clauses.append(opt2_clause)
        
        return sub_clauses
    
    def _create_sub_clause(self, text: str, clause_id: str, clause_type: str) -> LogicalClause:
        """Create a sub-clause from extracted text"""
        return LogicalClause(
            id=clause_id,
            text=text,
            normalized_form=text.lower().strip(),
            predicate=self._extract_main_predicate(text),
            subjects=self._extract_subjects(text),
            objects=self._extract_objects(text),
            modifiers=[clause_type],
            polarity=not self._has_negation(text),
            certainty=self._estimate_certainty(text),
            temporal_marker=self._extract_temporal_marker(text),
            conditional_type=clause_type if 'conditional' in clause_type else None,
            quantifier=self._detect_quantifier(text),
            sentence_position=int(clause_id.split('_')[0]) if '_' in clause_id else 0
        )
    
    def _extract_main_predicate(self, sentence: str) -> str:
        """Extract the main predicate (verb) from a sentence"""
        # Look for common verb patterns
        verb_patterns = [
            r'\b(is|are|was|were|being|been)\s+(\w+)',  # Copular verbs
            r'\b(\w+(?:s|ed|ing))\b',  # Regular verb forms
            r'\b(has|have|had)\s+(\w+)',  # Perfect tenses
            r'\b(will|would|can|could|should|must|might|may)\s+(\w+)',  # Modal verbs
        ]
        
        for pattern in verb_patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                if len(match.groups()) > 1:
                    return match.group(2)
                else:
                    return match.group(1)
        
        # Fallback: look for any word that might be a verb
        words = sentence.split()
        for word in words:
            if word.lower() in ['is', 'are', 'was', 'were', 'have', 'has', 'had', 'will', 'would', 'can', 'could']:
                return word.lower()
            elif word.endswith(('s', 'ed', 'ing')) and len(word) > 3:
                return word.lower()
        
        return "relates"  # Default predicate
    
    def _extract_subjects(self, sentence: str) -> List[str]:
        """Extract subject entities from sentence"""
        subjects = []
        
        # Look for capitalized words (proper nouns)
        capitalized = re.findall(r'\b[A-Z][a-z]+\b', sentence)
        subjects.extend(capitalized)
        
        # Look for pronouns at sentence start
        pronoun_match = re.match(r'^(I|You|He|She|It|We|They|This|That)\b', sentence, re.IGNORECASE)
        if pronoun_match:
            subjects.append(pronoun_match.group(1).lower())
        
        # Look for noun phrases before verbs
        noun_phrase_pattern = r'\b(the\s+\w+(?:\s+\w+)*|a\s+\w+(?:\s+\w+)*|\w+(?:\s+\w+)*)\s+(?:is|are|was|were|has|have|had|will|would|can|could|should|must|might|may)'
        matches = re.findall(noun_phrase_pattern, sentence, re.IGNORECASE)
        subjects.extend([match.strip() for match in matches])
        
        return list(set(subjects))  # Remove duplicates
    
    def _extract_objects(self, sentence: str) -> List[str]:
        """Extract object entities from sentence"""
        objects = []
        
        # Look for objects after verbs
        object_patterns = [
            r'(?:is|are|was|were)\s+(.*?)(?:\.|,|$)',
            r'(?:has|have|had)\s+(.*?)(?:\.|,|$)',
            r'(?:causes?|leads?\s+to|results?\s+in)\s+(.*?)(?:\.|,|$)'
        ]
        
        for pattern in object_patterns:
            matches = re.findall(pattern, sentence, re.IGNORECASE)
            for match in matches:
                if len(match.strip()) > 2:
                    objects.append(match.strip())
        
        return objects
    
    def _extract_modifiers(self, sentence: str) -> List[str]:
        """Extract modifying phrases and clauses"""
        modifiers = []
        
        # Look for adverbial phrases
        adverbial_patterns = [
            r'\b(very|extremely|highly|completely|totally|absolutely|quite|rather|fairly)\s+\w+',
            r'\b(in|on|at|by|with|through|during|after|before)\s+\w+(?:\s+\w+)*',
            r'\b\w+ly\b'  # Adverbs
        ]
        
        for pattern in adverbial_patterns:
            matches = re.findall(pattern, sentence, re.IGNORECASE)
            modifiers.extend(matches)
        
        return modifiers
    
    def _has_negation(self, sentence: str) -> bool:
        """Check if sentence contains negation"""
        for pattern in self.negation_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                return True
        return False
    
    def _estimate_certainty(self, sentence: str) -> float:
        """Estimate the certainty level of a statement"""
        sentence_lower = sentence.lower()
        
        # Check for high certainty markers
        for marker in self.certainty_markers['high']:
            if marker in sentence_lower:
                return 0.9
        
        # Check for medium certainty markers
        for marker in self.certainty_markers['medium']:
            if marker in sentence_lower:
                return 0.6
        
        # Check for low certainty markers
        for marker in self.certainty_markers['low']:
            if marker in sentence_lower:
                return 0.3
        
        # Default certainty for statements
        return 0.7
    
    def _extract_temporal_marker(self, sentence: str) -> Optional[str]:
        """Extract temporal markers from sentence"""
        temporal_patterns = [
            r'\b(before|after|during|while|when|whenever|until|since|now|then|today|yesterday|tomorrow)\b',
            r'\b(always|never|sometimes|often|rarely|frequently|occasionally|usually|typically)\b'
        ]
        
        for pattern in temporal_patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                return match.group(1).lower()
        
        return None
    
    def _detect_conditional_type(self, sentence: str) -> Optional[str]:
        """Detect type of conditional statement"""
        sentence_lower = sentence.lower()
        
        if 'if' in sentence_lower and 'then' in sentence_lower:
            return 'if_then'
        elif 'if' in sentence_lower:
            return 'if_implicit'
        elif 'when' in sentence_lower:
            return 'when'
        elif 'unless' in sentence_lower:
            return 'unless'
        elif 'provided' in sentence_lower or 'given that' in sentence_lower:
            return 'provided'
        elif 'assuming' in sentence_lower:
            return 'assuming'
        
        return None
    
    def _detect_quantifier(self, sentence: str) -> Optional[str]:
        """Detect quantifiers in sentence"""
        sentence_lower = sentence.lower()
        
        for quant_type, markers in self.quantifier_patterns.items():
            for marker in markers:
                if marker in sentence_lower:
                    return f"{quant_type}:{marker}"
        
        return None

class FormalArgumentDetector:
    """Detects formal logical arguments using the extracted clauses"""
    
    def __init__(self):
        self.pattern_extractor = LogicalPatternExtractor()
    
    def detect_arguments(self, text: str) -> List[LogicalArgument]:
        """Detect logical arguments in text"""
        clauses = self.pattern_extractor.extract_logical_clauses(text)
        arguments = []
        
        # Detect different argument types
        arguments.extend(self._detect_modus_ponens(clauses, text))
        arguments.extend(self._detect_modus_tollens(clauses, text))
        arguments.extend(self._detect_disjunctive_syllogism(clauses, text))
        arguments.extend(self._detect_hypothetical_syllogism(clauses, text))
        arguments.extend(self._detect_elimination_arguments(clauses, text))
        arguments.extend(self._detect_abductive_arguments(clauses, text))
        arguments.extend(self._detect_analogical_arguments(clauses, text))
        arguments.extend(self._detect_causal_arguments(clauses, text))
        arguments.extend(self._detect_reductio_arguments(clauses, text))
        arguments.extend(self._detect_confirmation_arguments(clauses, text))
        
        return arguments
    
    def _detect_modus_ponens(self, clauses: List[LogicalClause], text: str) -> List[LogicalArgument]:
        """Detect modus ponens: If P then Q, P, therefore Q"""
        arguments = []
        
        # Find conditional statements
        conditionals = [c for c in clauses if c.conditional_type and 'conditional' in c.conditional_type]
        
        for i, conditional in enumerate(conditionals):
            # Look for antecedent affirmation
            antecedents = [c for c in clauses if 'conditional_antecedent' in c.modifiers]
            consequents = [c for c in clauses if 'conditional_consequent' in c.modifiers]
            
            if antecedents and consequents:
                for ant in antecedents:
                    for cons in consequents:
                        # Look for a clause that matches the consequent (conclusion)
                        for clause in clauses:
                            if (clause.sentence_position > max(ant.sentence_position, cons.sentence_position) and
                                self._semantic_similarity(clause, cons) > 0.7):
                                
                                logical_form = f"({ant.normalized_form} → {cons.normalized_form}) ∧ {ant.normalized_form} ⊢ {cons.normalized_form}"
                                
                                arguments.append(LogicalArgument(
                                    id=f"MP_{len(arguments)}",
                                    argument_type=ArgumentType.MODUS_PONENS,
                                    premises=[conditional, ant],
                                    conclusion=clause,
                                    intermediate_steps=[],
                                    logical_form=logical_form,
                                    natural_language_form=f"If {ant.text}, then {cons.text}. {ant.text}. Therefore, {clause.text}",
                                    validity=True,
                                    soundness_estimate=min(ant.certainty, cons.certainty, clause.certainty),
                                    confidence=0.8,
                                    text_span=(min(conditional.sentence_position, ant.sentence_position), clause.sentence_position),
                                    inference_rules_used=["modus_ponens"]
                                ))
        
        return arguments
    
    def _detect_modus_tollens(self, clauses: List[LogicalClause], text: str) -> List[LogicalArgument]:
        """Detect modus tollens: If P then Q, not Q, therefore not P"""
        arguments = []
        
        conditionals = [c for c in clauses if c.conditional_type and 'conditional' in c.conditional_type]
        
        for conditional in conditionals:
            antecedents = [c for c in clauses if 'conditional_antecedent' in c.modifiers]
            consequents = [c for c in clauses if 'conditional_consequent' in c.modifiers]
            
            if antecedents and consequents:
                for ant in antecedents:
                    for cons in consequents:
                        # Look for negation of consequent
                        for clause in clauses:
                            if (clause.sentence_position > cons.sentence_position and
                                self._semantic_similarity(clause, cons) > 0.7 and
                                clause.polarity != cons.polarity):
                                
                                # Look for conclusion (negation of antecedent)
                                for conclusion in clauses:
                                    if (conclusion.sentence_position > clause.sentence_position and
                                        self._semantic_similarity(conclusion, ant) > 0.7 and
                                        conclusion.polarity != ant.polarity):
                                        
                                        logical_form = f"({ant.normalized_form} → {cons.normalized_form}) ∧ ¬{cons.normalized_form} ⊢ ¬{ant.normalized_form}"
                                        
                                        arguments.append(LogicalArgument(
                                            id=f"MT_{len(arguments)}",
                                            argument_type=ArgumentType.MODUS_TOLLENS,
                                            premises=[conditional, clause],
                                            conclusion=conclusion,
                                            intermediate_steps=[],
                                            logical_form=logical_form,
                                            natural_language_form=f"If {ant.text}, then {cons.text}. Not {cons.text}. Therefore, not {ant.text}",
                                            validity=True,
                                            soundness_estimate=min(ant.certainty, cons.certainty, clause.certainty),
                                            confidence=0.8,
                                            text_span=(conditional.sentence_position, conclusion.sentence_position),
                                            inference_rules_used=["modus_tollens"]
                                        ))
        
        return arguments
    
    def _detect_disjunctive_syllogism(self, clauses: List[LogicalClause], text: str) -> List[LogicalArgument]:
        """Detect disjunctive syllogism: P or Q, not P, therefore Q"""
        arguments = []
        
        # Find disjunctive statements
        disjunctives = [c for c in clauses if 'disjunctive_option' in c.modifiers]
        
        # Group disjunctive options that appear together
        disjunctive_pairs = []
        for i, clause1 in enumerate(disjunctives):
            for clause2 in disjunctives[i+1:]:
                if abs(clause1.sentence_position - clause2.sentence_position) <= 1:
                    disjunctive_pairs.append((clause1, clause2))
        
        for option1, option2 in disjunctive_pairs:
            # Look for negation of one option
            for clause in clauses:
                if clause.sentence_position > max(option1.sentence_position, option2.sentence_position):
                    if (self._semantic_similarity(clause, option1) > 0.7 and clause.polarity != option1.polarity):
                        # Negated option1, so conclusion should be option2
                        for conclusion in clauses:
                            if (conclusion.sentence_position > clause.sentence_position and
                                self._semantic_similarity(conclusion, option2) > 0.7):
                                
                                logical_form = f"({option1.normalized_form} ∨ {option2.normalized_form}) ∧ ¬{option1.normalized_form} ⊢ {option2.normalized_form}"
                                
                                arguments.append(LogicalArgument(
                                    id=f"DS_{len(arguments)}",
                                    argument_type=ArgumentType.DISJUNCTIVE_SYLLOGISM,
                                    premises=[option1, option2, clause],
                                    conclusion=conclusion,
                                    intermediate_steps=[],
                                    logical_form=logical_form,
                                    natural_language_form=f"Either {option1.text} or {option2.text}. Not {option1.text}. Therefore, {option2.text}",
                                    validity=True,
                                    soundness_estimate=min(option1.certainty, option2.certainty, clause.certainty),
                                    confidence=0.7,
                                    text_span=(min(option1.sentence_position, option2.sentence_position), conclusion.sentence_position),
                                    inference_rules_used=["disjunctive_syllogism"]
                                ))
                    
                    elif (self._semantic_similarity(clause, option2) > 0.7 and clause.polarity != option2.polarity):
                        # Negated option2, so conclusion should be option1
                        for conclusion in clauses:
                            if (conclusion.sentence_position > clause.sentence_position and
                                self._semantic_similarity(conclusion, option1) > 0.7):
                                
                                logical_form = f"({option1.normalized_form} ∨ {option2.normalized_form}) ∧ ¬{option2.normalized_form} ⊢ {option1.normalized_form}"
                                
                                arguments.append(LogicalArgument(
                                    id=f"DS_{len(arguments)}",
                                    argument_type=ArgumentType.DISJUNCTIVE_SYLLOGISM,
                                    premises=[option1, option2, clause],
                                    conclusion=conclusion,
                                    intermediate_steps=[],
                                    logical_form=logical_form,
                                    natural_language_form=f"Either {option1.text} or {option2.text}. Not {option2.text}. Therefore, {option1.text}",
                                    validity=True,
                                    soundness_estimate=min(option1.certainty, option2.certainty, clause.certainty),
                                    confidence=0.7,
                                    text_span=(min(option1.sentence_position, option2.sentence_position), conclusion.sentence_position),
                                    inference_rules_used=["disjunctive_syllogism"]
                                ))
        
        return arguments
    
    def _detect_elimination_arguments(self, clauses: List[LogicalClause], text: str) -> List[LogicalArgument]:
        """Detect elimination arguments: process of eliminating options"""
        arguments = []
        
        # Look for patterns where multiple options are considered and eliminated
        eliminated_options = []
        remaining_options = []
        option_statements = []
        
        for clause in clauses:
            # Look for option introduction patterns
            if any(phrase in clause.text.lower() for phrase in 
                   ['the first thought', 'first possibility', 'then i think about', 'now the third possibility', 
                    'the fourth option', 'another possibility', 'one option', 'another thought']):
                option_statements.append(clause)
            
            # Look for elimination language (more comprehensive)
            elif any(phrase in clause.text.lower() for phrase in 
                   ['that can\'t be right', 'that doesn\'t make sense', 'that seems unlikely', 
                    'i can rule out', 'that\'s not possible', 'that\'s incorrect', 'that immediately strikes me as wrong',
                    'that\'s actually contradicting', 'doesn\'t seem right', 'that can\'t be the answer',
                    'this doesn\'t make any sense', 'that would be wrong', 'this is wrong']):
                eliminated_options.append(clause)
            
            # Look for confirmation language (more comprehensive)
            elif any(phrase in clause.text.lower() for phrase in 
                     ['that makes sense', 'that\'s correct', 'that\'s the right answer', 
                      'i\'m confident', 'that must be it', 'that\'s really the key', 
                      'i keep coming back to', 'that\'s exactly what', 'this is making me think',
                      'really catches my attention', 'the answer is']):
                remaining_options.append(clause)
        
        # Create elimination arguments from patterns
        if eliminated_options and remaining_options:
            for remaining in remaining_options:
                # Find relevant eliminated options (those that appear before the conclusion)
                relevant_eliminated = [e for e in eliminated_options 
                                     if e.sentence_position < remaining.sentence_position]
                
                if relevant_eliminated:
                    premises = relevant_eliminated[:4]  # Use up to 4 elimination premises
                    
                    logical_form = f"¬P₁ ∧ ¬P₂ ∧ ... ∧ (P₁ ∨ P₂ ∨ ... ∨ Pₙ) ⊢ Pₙ"
                    
                    # Create natural language form
                    eliminated_descriptions = []
                    for premise in premises:
                        if 'contradicting' in premise.text.lower():
                            eliminated_descriptions.append('ruled out due to contradiction')
                        elif 'wrong' in premise.text.lower():
                            eliminated_descriptions.append('determined to be incorrect')
                        elif 'doesn\'t make sense' in premise.text.lower():
                            eliminated_descriptions.append('rejected as nonsensical')
                        else:
                            eliminated_descriptions.append('eliminated')
                    
                    natural_form = f"After eliminating options ({', '.join(eliminated_descriptions)}), concluded: {remaining.text}"
                    
                    arguments.append(LogicalArgument(
                        id=f"ELIM_{len(arguments)}",
                        argument_type=ArgumentType.ELIMINATION,
                        premises=premises,
                        conclusion=remaining,
                        intermediate_steps=[],
                        logical_form=logical_form,
                        natural_language_form=natural_form,
                        validity=True,
                        soundness_estimate=min(remaining.certainty, max(p.certainty for p in premises)),
                        confidence=0.8,
                        text_span=(min(p.sentence_position for p in premises), remaining.sentence_position),
                        inference_rules_used=["elimination"]
                    ))
        
        # Also look for systematic option analysis patterns
        if len(option_statements) >= 2 and remaining_options:
            for remaining in remaining_options:
                if remaining.sentence_position > max(opt.sentence_position for opt in option_statements):
                    # This appears to be a conclusion after systematic option analysis
                    logical_form = f"Consider(P₁, P₂, ..., Pₙ) ∧ Analyze(P₁, P₂, ..., Pₙ) ⊢ Best(Pₖ)"
                    
                    arguments.append(LogicalArgument(
                        id=f"OPT_ELIM_{len(arguments)}",
                        argument_type=ArgumentType.ELIMINATION,
                        premises=option_statements[:3],  # Use up to 3 option premises
                        conclusion=remaining,
                        intermediate_steps=eliminated_options,
                        logical_form=logical_form,
                        natural_language_form=f"After systematically considering multiple options, selected: {remaining.text}",
                        validity=True,
                        soundness_estimate=remaining.certainty * 0.9,
                        confidence=0.85,
                        text_span=(min(opt.sentence_position for opt in option_statements), remaining.sentence_position),
                        inference_rules_used=["systematic_elimination"]
                    ))
                    break  # Only create one systematic elimination argument
        
        return arguments
    
    def _detect_abductive_arguments(self, clauses: List[LogicalClause], text: str) -> List[LogicalArgument]:
        """Detect abductive reasoning: inference to the best explanation"""
        arguments = []
        
        # Look for explanation language (expanded patterns)
        explanation_clauses = []
        observation_clauses = []
        requirement_clauses = []
        
        for clause in clauses:
            if any(phrase in clause.text.lower() for phrase in 
                   ['best explanation', 'most likely', 'probably because', 'this explains',
                    'the reason is', 'this accounts for', 'this is due to', 'what would make it suitable',
                    'this is what makes', 'the key here', 'this is exactly what']):
                explanation_clauses.append(clause)
            elif any(phrase in clause.text.lower() for phrase in 
                     ['we observe', 'the evidence shows', 'what we see', 'the data indicates',
                      'the question mentions', 'known for', 'the fact that']):
                observation_clauses.append(clause)
            elif any(phrase in clause.text.lower() for phrase in 
                     ['we need', 'require', 'absolutely need', 'must have', 'we want',
                      'for permanent magnet', 'suitable for']):
                requirement_clauses.append(clause)
        
        # Pair observations with explanations
        for explanation in explanation_clauses:
            relevant_observations = [obs for obs in observation_clauses 
                                   if obs.sentence_position < explanation.sentence_position]
            
            if relevant_observations:
                logical_form = f"Obs₁ ∧ Obs₂ ∧ ... ∧ (H best explains Obs) ⊢ H"
                
                arguments.append(LogicalArgument(
                    id=f"ABD_{len(arguments)}",
                    argument_type=ArgumentType.ABDUCTIVE,
                    premises=relevant_observations,
                    conclusion=explanation,
                    intermediate_steps=[],
                    logical_form=logical_form,
                    natural_language_form=f"Given observations, {explanation.text} is the best explanation",
                    validity=False,  # Abductive reasoning is not deductively valid
                    soundness_estimate=explanation.certainty * 0.7,  # Reduce for non-deductive
                    confidence=0.6,
                    text_span=(min(obs.sentence_position for obs in relevant_observations), explanation.sentence_position),
                    inference_rules_used=["abduction"]
                ))
        
        # Look for requirement-based abductive reasoning
        # Pattern: "For X we need Y, Z has Y, therefore Z is suitable for X"
        for requirement in requirement_clauses:
            for explanation in explanation_clauses:
                if (explanation.sentence_position > requirement.sentence_position and
                    abs(explanation.sentence_position - requirement.sentence_position) <= 5):
                    
                    # Check if there are property statements between requirement and explanation
                    property_statements = [c for c in clauses 
                                         if (c.sentence_position > requirement.sentence_position and 
                                             c.sentence_position < explanation.sentence_position and
                                             any(prop in c.text.lower() for prop in 
                                                 ['properties', 'structure', 'arrangement', 'magnetic', 'ordered']))]
                    
                    if property_statements:
                        premises = [requirement] + property_statements[:2]  # Requirement + up to 2 property statements
                        
                        logical_form = f"Required(R) ∧ HasProperty(X, P) ∧ Enables(P, R) ⊢ Suitable(X, R)"
                        
                        arguments.append(LogicalArgument(
                            id=f"REQ_ABD_{len(arguments)}",
                            argument_type=ArgumentType.ABDUCTIVE,
                            premises=premises,
                            conclusion=explanation,
                            intermediate_steps=[],
                            logical_form=logical_form,
                            natural_language_form=f"Requirements: {requirement.text} → Properties match → Conclusion: {explanation.text}",
                            validity=False,  # Requirement-based reasoning is abductive
                            soundness_estimate=explanation.certainty * 0.8,
                            confidence=0.75,
                            text_span=(requirement.sentence_position, explanation.sentence_position),
                            inference_rules_used=["requirement_based_abduction"]
                        ))
        
        return arguments
    
    def _detect_analogical_arguments(self, clauses: List[LogicalClause], text: str) -> List[LogicalArgument]:
        """Detect analogical reasoning"""
        arguments = []
        
        # Look for analogy markers
        for clause in clauses:
            if any(phrase in clause.text.lower() for phrase in 
                   ['similar to', 'like', 'analogous to', 'just as', 'comparable to', 'resembles']):
                
                # Find what's being compared
                following_clauses = [c for c in clauses if c.sentence_position > clause.sentence_position]
                
                if following_clauses:
                    conclusion = following_clauses[0]  # Take the next clause as conclusion
                    
                    logical_form = f"A ≈ B ∧ A has property P ⊢ B probably has property P"
                    
                    arguments.append(LogicalArgument(
                        id=f"ANAL_{len(arguments)}",
                        argument_type=ArgumentType.ANALOGICAL,
                        premises=[clause],
                        conclusion=conclusion,
                        intermediate_steps=[],
                        logical_form=logical_form,
                        natural_language_form=f"Based on analogy: {clause.text}, therefore {conclusion.text}",
                        validity=False,  # Analogical reasoning is not deductively valid
                        soundness_estimate=min(clause.certainty, conclusion.certainty) * 0.6,
                        confidence=0.5,
                        text_span=(clause.sentence_position, conclusion.sentence_position),
                        inference_rules_used=["analogical_reasoning"]
                    ))
        
        return arguments
    
    def _detect_causal_arguments(self, clauses: List[LogicalClause], text: str) -> List[LogicalArgument]:
        """Detect causal reasoning"""
        arguments = []
        
        # Find causal clauses from patterns
        causal_antecedents = [c for c in clauses if 'causal_antecedent' in c.modifiers]
        causal_consequents = [c for c in clauses if 'causal_consequent' in c.modifiers]
        
        # Also detect causal patterns directly in text
        causal_statements = []
        property_statements = []
        
        for clause in clauses:
            # Look for explicit causal language
            if any(phrase in clause.text.lower() for phrase in 
                   ['this can create', 'this would create', 'this gives', 'this would give',
                    'this makes', 'this causes', 'this leads to', 'this results in',
                    'this accounts for', 'this explains why', 'because of this']):
                causal_statements.append(clause)
            
            # Look for property/structure statements
            elif any(phrase in clause.text.lower() for phrase in 
                     ['crystal structure', 'atomic arrangement', 'ordered structure', 
                      'magnetic properties', 'ferromagnetic', 'ordered arrangement',
                      'structural', 'properties', 'arrangement']):
                property_statements.append(clause)
        
        # Match traditional causal clauses
        for cause in causal_antecedents:
            for effect in causal_consequents:
                if abs(cause.sentence_position - effect.sentence_position) <= 2:
                    logical_form = f"Cause({cause.normalized_form}) ⊢ Effect({effect.normalized_form})"
                    
                    arguments.append(LogicalArgument(
                        id=f"CAUS_{len(arguments)}",
                        argument_type=ArgumentType.CAUSAL,
                        premises=[cause],
                        conclusion=effect,
                        intermediate_steps=[],
                        logical_form=logical_form,
                        natural_language_form=f"{cause.text} causes {effect.text}",
                        validity=False,  # Causal reasoning can be fallible
                        soundness_estimate=min(cause.certainty, effect.certainty) * 0.8,
                        confidence=0.7,
                        text_span=(cause.sentence_position, effect.sentence_position),
                        inference_rules_used=["causal_reasoning"]
                    ))
        
        # Match causal statements with property statements
        for causal in causal_statements:
            # Find related property statements that appear before this causal statement
            relevant_properties = [p for p in property_statements 
                                 if p.sentence_position < causal.sentence_position 
                                 and abs(p.sentence_position - causal.sentence_position) <= 3]
            
            if relevant_properties:
                # Use the closest property statement as the premise
                closest_property = max(relevant_properties, key=lambda x: x.sentence_position)
                
                logical_form = f"Structure({closest_property.normalized_form}) ⊢ Property({causal.normalized_form})"
                
                arguments.append(LogicalArgument(
                    id=f"STRUCT_CAUS_{len(arguments)}",
                    argument_type=ArgumentType.CAUSAL,
                    premises=[closest_property],
                    conclusion=causal,
                    intermediate_steps=[],
                    logical_form=logical_form,
                    natural_language_form=f"Structure/arrangement: {closest_property.text} → Properties: {causal.text}",
                    validity=False,  # Structure-property relationships are empirical
                    soundness_estimate=min(closest_property.certainty, causal.certainty) * 0.85,
                    confidence=0.8,
                    text_span=(closest_property.sentence_position, causal.sentence_position),
                    inference_rules_used=["structure_property_reasoning"]
                ))
        
        return arguments
    
    def _detect_hypothetical_syllogism(self, clauses: List[LogicalClause], text: str) -> List[LogicalArgument]:
        """Detect hypothetical syllogism: If P then Q, if Q then R, therefore if P then R"""
        arguments = []
        
        # This is complex and would require tracking conditional chains
        # Simplified implementation for now
        conditionals = [c for c in clauses if c.conditional_type]
        
        if len(conditionals) >= 2:
            for i, cond1 in enumerate(conditionals):
                for cond2 in conditionals[i+1:]:
                    # Simplified detection - would need more sophisticated matching
                    logical_form = f"(P → Q) ∧ (Q → R) ⊢ (P → R)"
                    
                    arguments.append(LogicalArgument(
                        id=f"HS_{len(arguments)}",
                        argument_type=ArgumentType.HYPOTHETICAL_SYLLOGISM,
                        premises=[cond1, cond2],
                        conclusion=cond2,  # Simplified
                        intermediate_steps=[],
                        logical_form=logical_form,
                        natural_language_form=f"Chain of conditionals: {cond1.text} and {cond2.text}",
                        validity=True,
                        soundness_estimate=min(cond1.certainty, cond2.certainty),
                        confidence=0.6,
                        text_span=(cond1.sentence_position, cond2.sentence_position),
                        inference_rules_used=["hypothetical_syllogism"]
                    ))
                    break  # Only create one per pair
        
        return arguments
    
    def _detect_reductio_arguments(self, clauses: List[LogicalClause], text: str) -> List[LogicalArgument]:
        """Detect reductio ad absurdum arguments"""
        arguments = []
        
        # Look for assumption language followed by contradiction language
        assumptions = []
        contradictions = []
        conclusions = []
        
        for clause in clauses:
            if any(phrase in clause.text.lower() for phrase in 
                   ['assume', 'suppose', 'let\'s say', 'if we consider', 'hypothetically']):
                assumptions.append(clause)
            elif any(phrase in clause.text.lower() for phrase in 
                     ['contradiction', 'impossible', 'can\'t be', 'doesn\'t make sense', 'absurd']):
                contradictions.append(clause)
            elif any(phrase in clause.text.lower() for phrase in 
                     ['therefore', 'thus', 'so', 'we can conclude']):
                conclusions.append(clause)
        
        # Match assumptions with contradictions and conclusions
        for assumption in assumptions:
            relevant_contradictions = [c for c in contradictions 
                                     if c.sentence_position > assumption.sentence_position]
            relevant_conclusions = [c for c in conclusions 
                                  if c.sentence_position > assumption.sentence_position]
            
            if relevant_contradictions and relevant_conclusions:
                contradiction = relevant_contradictions[0]
                conclusion = relevant_conclusions[0]
                
                logical_form = f"Assume ¬P ⊢ ⊥, therefore P"
                
                arguments.append(LogicalArgument(
                    id=f"RAA_{len(arguments)}",
                    argument_type=ArgumentType.REDUCTIO_AD_ABSURDUM,
                    premises=[assumption, contradiction],
                    conclusion=conclusion,
                    intermediate_steps=[],
                    logical_form=logical_form,
                    natural_language_form=f"Assumed {assumption.text}, led to contradiction, therefore {conclusion.text}",
                    validity=True,
                    soundness_estimate=min(assumption.certainty, conclusion.certainty),
                    confidence=0.8,
                    text_span=(assumption.sentence_position, conclusion.sentence_position),
                    inference_rules_used=["reductio_ad_absurdum"]
                ))
        
        return arguments
    
    def _detect_confirmation_arguments(self, clauses: List[LogicalClause], text: str) -> List[LogicalArgument]:
        """Detect confirmation/support arguments"""
        arguments = []
        
        # Look for evidence supporting conclusions
        evidence_clauses = []
        conclusion_clauses = []
        
        for clause in clauses:
            if any(phrase in clause.text.lower() for phrase in 
                   ['evidence shows', 'research indicates', 'studies demonstrate', 'data supports']):
                evidence_clauses.append(clause)
            elif any(phrase in clause.text.lower() for phrase in 
                     ['confirms', 'supports', 'validates', 'proves', 'demonstrates']):
                conclusion_clauses.append(clause)
        
        for evidence in evidence_clauses:
            relevant_conclusions = [c for c in conclusion_clauses 
                                  if c.sentence_position > evidence.sentence_position]
            
            if relevant_conclusions:
                conclusion = relevant_conclusions[0]
                
                logical_form = f"Evidence(E) ∧ E supports H ⊢ H is more likely"
                
                arguments.append(LogicalArgument(
                    id=f"CONF_{len(arguments)}",
                    argument_type=ArgumentType.CONFIRMATION,
                    premises=[evidence],
                    conclusion=conclusion,
                    intermediate_steps=[],
                    logical_form=logical_form,
                    natural_language_form=f"Evidence supports conclusion: {evidence.text} therefore {conclusion.text}",
                    validity=False,  # Confirmatory evidence doesn't guarantee truth
                    soundness_estimate=min(evidence.certainty, conclusion.certainty) * 0.8,
                    confidence=0.7,
                    text_span=(evidence.sentence_position, conclusion.sentence_position),
                    inference_rules_used=["confirmation"]
                ))
        
        return arguments
    
    def _semantic_similarity(self, clause1: LogicalClause, clause2: LogicalClause) -> float:
        """Calculate semantic similarity between clauses"""
        # Simplified similarity based on word overlap
        words1 = set(clause1.normalized_form.split())
        words2 = set(clause2.normalized_form.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # Boost similarity if predicates match
        if clause1.predicate == clause2.predicate:
            jaccard += 0.3
        
        # Boost similarity if subjects overlap
        subject_overlap = set(clause1.subjects).intersection(set(clause2.subjects))
        if subject_overlap:
            jaccard += 0.2
        
        return min(1.0, jaccard)

class EnhancedFormalLogicAnalyzer:
    """Main analyzer class"""
    
    def __init__(self):
        self.argument_detector = FormalArgumentDetector()
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a single stream file for formal logic structures"""
        stream_content = self._extract_stream_content(file_path)
        if not stream_content:
            return self._empty_analysis(file_path)
        
        # Detect logical arguments
        arguments = self.argument_detector.detect_arguments(stream_content)
        
        # Perform analysis
        analysis = {
            'file_path': file_path,
            'text_length': len(stream_content),
            'total_arguments': len(arguments),
            'argument_types': self._categorize_arguments(arguments),
            'deductive_arguments': [arg.to_dict() if hasattr(arg, 'to_dict') else self._argument_to_dict(arg) 
                                  for arg in arguments if arg.validity],
            'non_deductive_arguments': [arg.to_dict() if hasattr(arg, 'to_dict') else self._argument_to_dict(arg) 
                                      for arg in arguments if not arg.validity],
            'logical_complexity': self._calculate_complexity_metrics(arguments),
            'argument_quality': self._assess_argument_quality(arguments),
            'inference_patterns': self._analyze_inference_patterns(arguments),
            'formal_representations': [arg.logical_form for arg in arguments],
            'natural_language_forms': [arg.natural_language_form for arg in arguments],
            'argument_chains': self._detect_argument_chains(arguments),
            'validity_assessment': self._assess_overall_validity(arguments)
        }
        
        return analysis
    
    def _extract_stream_content(self, file_path: str) -> str:
        """Extract stream content from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find stream section
            stream_start = content.find("STREAM OF THOUGHT:")
            if stream_start == -1:
                return ""
            
            stream_end = content.find("=" * 80, stream_start + 1)
            if stream_end == -1:
                stream_content = content[stream_start:]
            else:
                stream_content = content[stream_start:stream_end]
            
            # Clean up
            lines = stream_content.split('\n')
            clean_lines = []
            skip_header = True
            
            for line in lines:
                if skip_header and ("STREAM OF THOUGHT:" in line or line.strip() == "-" * 40):
                    continue
                skip_header = False
                clean_lines.append(line)
            
            return '\n'.join(clean_lines).strip()
        except Exception:
            return ""
    
    def _empty_analysis(self, file_path: str) -> Dict[str, Any]:
        """Return empty analysis structure"""
        return {
            'file_path': file_path,
            'text_length': 0,
            'total_arguments': 0,
            'argument_types': {},
            'deductive_arguments': [],
            'non_deductive_arguments': [],
            'logical_complexity': {
                'argument_density': 0.0,
                'avg_argument_strength': 0.0,
                'deductive_ratio': 0.0,
                'inference_diversity': 0.0
            },
            'argument_quality': {
                'avg_soundness': 0.0,
                'avg_confidence': 0.0,
                'validity_rate': 0.0
            },
            'inference_patterns': {},
            'formal_representations': [],
            'natural_language_forms': [],
            'argument_chains': [],
            'validity_assessment': {
                'overall_validity': 0.0,
                'logical_consistency': 0.0,
                'argument_strength': 0.0
            }
        }
    
    def _argument_to_dict(self, arg: LogicalArgument) -> Dict[str, Any]:
        """Convert argument to dictionary"""
        return {
            'id': arg.id,
            'argument_type': arg.argument_type.value,
            'premises': [clause.to_dict() for clause in arg.premises],
            'conclusion': arg.conclusion.to_dict(),
            'logical_form': arg.logical_form,
            'natural_language_form': arg.natural_language_form,
            'validity': arg.validity,
            'soundness_estimate': arg.soundness_estimate,
            'confidence': arg.confidence,
            'text_span': arg.text_span,
            'inference_rules_used': arg.inference_rules_used
        }
    
    def _categorize_arguments(self, arguments: List[LogicalArgument]) -> Dict[str, int]:
        """Categorize arguments by type"""
        categories = defaultdict(int)
        for arg in arguments:
            categories[arg.argument_type.value] += 1
        return dict(categories)
    
    def _calculate_complexity_metrics(self, arguments: List[LogicalArgument]) -> Dict[str, float]:
        """Calculate complexity metrics"""
        if not arguments:
            return {
                'argument_density': 0.0,
                'avg_argument_strength': 0.0,
                'deductive_ratio': 0.0,
                'inference_diversity': 0.0
            }
        
        deductive_count = sum(1 for arg in arguments if arg.validity)
        total_args = len(arguments)
        
        return {
            'argument_density': total_args,  # Could normalize by text length
            'avg_argument_strength': sum(arg.soundness_estimate for arg in arguments) / total_args,
            'deductive_ratio': deductive_count / total_args,
            'inference_diversity': len(set(arg.argument_type for arg in arguments)) / len(ArgumentType)
        }
    
    def _assess_argument_quality(self, arguments: List[LogicalArgument]) -> Dict[str, float]:
        """Assess overall argument quality"""
        if not arguments:
            return {
                'avg_soundness': 0.0,
                'avg_confidence': 0.0,
                'validity_rate': 0.0
            }
        
        return {
            'avg_soundness': sum(arg.soundness_estimate for arg in arguments) / len(arguments),
            'avg_confidence': sum(arg.confidence for arg in arguments) / len(arguments),
            'validity_rate': sum(1 for arg in arguments if arg.validity) / len(arguments)
        }
    
    def _analyze_inference_patterns(self, arguments: List[LogicalArgument]) -> Dict[str, int]:
        """Analyze patterns in inference rules used"""
        patterns = defaultdict(int)
        for arg in arguments:
            for rule in arg.inference_rules_used:
                patterns[rule] += 1
        return dict(patterns)
    
    def _detect_argument_chains(self, arguments: List[LogicalArgument]) -> List[Dict[str, Any]]:
        """Detect chains of connected arguments"""
        chains = []
        
        for i, arg1 in enumerate(arguments):
            for j, arg2 in enumerate(arguments[i+1:], i+1):
                # Check if conclusion of arg1 overlaps with premises of arg2
                conclusion_text = arg1.conclusion.normalized_form
                
                for premise in arg2.premises:
                    premise_text = premise.normalized_form
                    
                    # Simple overlap check
                    words1 = set(conclusion_text.split())
                    words2 = set(premise_text.split())
                    overlap = len(words1.intersection(words2))
                    
                    if overlap >= 2:  # At least 2 words in common
                        chains.append({
                            'from_argument': arg1.id,
                            'to_argument': arg2.id,
                            'connection_strength': overlap / max(len(words1), len(words2)),
                            'chain_types': [arg1.argument_type.value, arg2.argument_type.value]
                        })
                        break
        
        return chains
    
    def _assess_overall_validity(self, arguments: List[LogicalArgument]) -> Dict[str, float]:
        """Assess overall logical validity"""
        if not arguments:
            return {
                'overall_validity': 0.0,
                'logical_consistency': 0.0,
                'argument_strength': 0.0
            }
        
        valid_args = [arg for arg in arguments if arg.validity]
        validity_rate = len(valid_args) / len(arguments)
        
        avg_confidence = sum(arg.confidence for arg in arguments) / len(arguments)
        avg_soundness = sum(arg.soundness_estimate for arg in arguments) / len(arguments)
        
        return {
            'overall_validity': validity_rate,
            'logical_consistency': avg_confidence,
            'argument_strength': avg_soundness
        }

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced formal logic analysis of reasoning structures")
    parser.add_argument("directory", help="Directory containing stream analysis files")
    parser.add_argument("--output", "-o", default="enhanced_formal_logic_analysis.json",
                       help="Output file for analysis results")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.directory):
        print(f"Error: Directory {args.directory} does not exist")
        return
    
    analyzer = EnhancedFormalLogicAnalyzer()
    results = {}
    
    # Find stream analysis files
    import glob
    pattern = os.path.join(args.directory, "*_STREAM_ANALYSIS.txt")
    files = glob.glob(pattern)
    
    print(f"Analyzing {len(files)} stream analysis files...")
    
    for file_path in files:
        filename = os.path.basename(file_path)
        print(f"Processing {filename}...")
        
        analysis = analyzer.analyze_file(file_path)
        results[filename] = analysis
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nAnalysis complete. Results saved to {args.output}")
    
    # Print summary
    if results:
        total_args = sum(result['total_arguments'] for result in results.values())
        total_deductive = sum(len(result['deductive_arguments']) for result in results.values())
        
        print(f"\nSUMMARY:")
        print(f"Total logical arguments detected: {total_args}")
        print(f"Deductive arguments: {total_deductive}")
        print(f"Non-deductive arguments: {total_args - total_deductive}")
        
        # Aggregate argument types
        all_types = defaultdict(int)
        for result in results.values():
            for arg_type, count in result['argument_types'].items():
                all_types[arg_type] += count
        
        if all_types:
            print(f"\nArgument types detected:")
            for arg_type, count in sorted(all_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  {arg_type.replace('_', ' ').title()}: {count}")
        
        # Average metrics
        if results:
            avg_validity = sum(r['validity_assessment']['overall_validity'] for r in results.values()) / len(results)
            avg_strength = sum(r['validity_assessment']['argument_strength'] for r in results.values()) / len(results)
            
            print(f"\nAverage validity rate: {avg_validity:.2f}")
            print(f"Average argument strength: {avg_strength:.2f}")

if __name__ == "__main__":
    main()