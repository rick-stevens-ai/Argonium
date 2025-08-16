#!/usr/bin/env python3
"""
Formal Logic Argument Structure Analyzer

This script analyzes the logical argument structure in Stream of Thought outputs
using formal logic principles (first-order and higher-order logic) rather than
keyword-based pattern matching.

It identifies:
1. Logical argument forms (modus ponens, modus tollens, disjunctive syllogism, etc.)
2. Proposition relationships and dependencies
3. Inference chains and logical validity
4. Argument structure types (deductive, inductive, abductive)
5. Logical fallacies and invalid reasoning patterns
"""

import re
import json
import os
from typing import Dict, List, Tuple, Set, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
from enum import Enum

# Try to import optional dependencies
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

class ArgumentType(Enum):
    """Types of logical arguments"""
    MODUS_PONENS = "modus_ponens"           # If P then Q, P, therefore Q
    MODUS_TOLLENS = "modus_tollens"         # If P then Q, not Q, therefore not P
    HYPOTHETICAL_SYLLOGISM = "hypothetical_syllogism"  # If P then Q, if Q then R, therefore if P then R
    DISJUNCTIVE_SYLLOGISM = "disjunctive_syllogism"   # P or Q, not P, therefore Q
    CONSTRUCTIVE_DILEMMA = "constructive_dilemma"     # (P→Q)∧(R→S)∧(P∨R) ⊢ Q∨S
    UNIVERSAL_INSTANTIATION = "universal_instantiation" # ∀x P(x), therefore P(a)
    EXISTENTIAL_GENERALIZATION = "existential_generalization" # P(a), therefore ∃x P(x)
    CONTRAPOSITION = "contraposition"       # P→Q is equivalent to ¬Q→¬P
    DE_MORGAN = "de_morgan"                # ¬(P∧Q) ≡ (¬P∨¬Q)
    REDUCTIO_AD_ABSURDUM = "reductio_ad_absurdum"  # Assume ¬P, derive contradiction, conclude P
    ABDUCTIVE = "abductive"                # Best explanation inference
    INDUCTIVE_GENERALIZATION = "inductive_generalization"  # Pattern-based generalization
    ANALOGICAL = "analogical"              # Reasoning by analogy
    CAUSAL = "causal"                      # Cause-effect reasoning
    PROBABILISTIC = "probabilistic"        # Likelihood-based reasoning

class LogicalOperator(Enum):
    """Logical operators"""
    AND = "∧"
    OR = "∨"
    NOT = "¬"
    IMPLIES = "→"
    IFF = "↔"
    FORALL = "∀"
    EXISTS = "∃"

@dataclass
class Proposition:
    """Represents a logical proposition"""
    id: str
    text: str
    predicate: str
    arguments: List[str]
    polarity: bool  # True for positive, False for negative
    modal_strength: float  # 0.0 (impossible) to 1.0 (certain)
    context_position: int
    
    def __hash__(self):
        return hash((self.predicate, tuple(self.arguments), self.polarity))

@dataclass
class LogicalRelation:
    """Represents a logical relationship between propositions"""
    relation_type: LogicalOperator
    antecedent: Optional[Proposition]
    consequent: Optional[Proposition]
    premises: List[Proposition]
    conclusion: Proposition
    confidence: float

@dataclass
class ArgumentInstance:
    """Represents an identified logical argument"""
    argument_type: ArgumentType
    premises: List[Proposition]
    conclusion: Proposition
    validity: bool
    confidence: float
    text_span: str
    logical_form: str

class PropositionExtractor:
    """Extracts propositions from natural language text"""
    
    def __init__(self):
        if HAS_SPACY:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("Warning: spaCy model 'en_core_web_sm' not found. Using fallback extraction.")
                self.nlp = None
        else:
            print("Warning: spaCy not available. Using fallback extraction.")
            self.nlp = None
    
    def extract_propositions(self, text: str) -> List[Proposition]:
        """Extract propositions from text using syntactic and semantic analysis"""
        if not self.nlp:
            return self._fallback_extraction(text)
        
        doc = self.nlp(text)
        propositions = []
        prop_id = 0
        
        # Extract propositions from sentences
        for sent_idx, sent in enumerate(doc.sents):
            # Find main clauses and subordinate clauses
            main_clauses = self._extract_clauses(sent)
            
            for clause in main_clauses:
                prop = self._clause_to_proposition(clause, prop_id, sent_idx)
                if prop:
                    propositions.append(prop)
                    prop_id += 1
        
        return propositions
    
    def _extract_clauses(self, sent) -> List[Dict]:
        """Extract clauses from a sentence"""
        clauses = []
        
        # Find root verb and its dependents
        root = None
        for token in sent:
            if token.dep_ == "ROOT":
                root = token
                break
        
        if root and root.pos_ == "VERB":
            # Main clause
            subject = self._find_subject(root)
            obj = self._find_object(root)
            
            clause = {
                'type': 'main',
                'verb': root,
                'subject': subject,
                'object': obj,
                'negation': self._has_negation(root),
                'modal': self._get_modal_strength(root),
                'tokens': list(sent)
            }
            clauses.append(clause)
            
            # Find subordinate clauses
            for child in root.children:
                if child.dep_ in ["advcl", "ccomp", "xcomp", "acl"]:
                    sub_clause = self._extract_subordinate_clause(child)
                    if sub_clause:
                        clauses.append(sub_clause)
        
        return clauses
    
    def _clause_to_proposition(self, clause: Dict, prop_id: int, position: int) -> Optional[Proposition]:
        """Convert a clause to a proposition"""
        verb = clause['verb']
        subject = clause.get('subject')
        obj = clause.get('object')
        
        if not subject:
            return None
        
        # Create predicate and arguments
        predicate = verb.lemma_
        arguments = []
        
        if subject:
            arguments.append(self._token_to_string(subject))
        if obj:
            arguments.append(self._token_to_string(obj))
        
        # Extract full text span
        text_tokens = clause['tokens']
        text = ' '.join([t.text for t in text_tokens])
        
        return Proposition(
            id=f"P{prop_id}",
            text=text,
            predicate=predicate,
            arguments=arguments,
            polarity=not clause['negation'],
            modal_strength=clause['modal'],
            context_position=position
        )
    
    def _find_subject(self, verb):
        """Find the subject of a verb"""
        for child in verb.children:
            if child.dep_ in ["nsubj", "nsubjpass", "csubj"]:
                return child
        return None
    
    def _find_object(self, verb):
        """Find the object of a verb"""
        for child in verb.children:
            if child.dep_ in ["dobj", "pobj", "iobj"]:
                return child
        return None
    
    def _has_negation(self, verb) -> bool:
        """Check if verb is negated"""
        for child in verb.children:
            if child.dep_ == "neg":
                return True
        return "not" in verb.text.lower() or "n't" in verb.text.lower()
    
    def _get_modal_strength(self, verb) -> float:
        """Estimate modal strength (certainty) of the verb"""
        # Look for modal auxiliaries
        for child in verb.children:
            if child.dep_ == "aux" and child.lemma_ in ["might", "could", "may"]:
                return 0.3
            elif child.dep_ == "aux" and child.lemma_ in ["should", "would"]:
                return 0.7
            elif child.dep_ == "aux" and child.lemma_ in ["must", "will"]:
                return 0.9
        
        # Default certainty for statements
        return 0.8
    
    def _extract_subordinate_clause(self, token) -> Optional[Dict]:
        """Extract subordinate clause information"""
        if token.pos_ == "VERB":
            subject = self._find_subject(token)
            obj = self._find_object(token)
            
            return {
                'type': 'subordinate',
                'verb': token,
                'subject': subject,
                'object': obj,
                'negation': self._has_negation(token),
                'modal': self._get_modal_strength(token),
                'tokens': [token] + list(token.subtree)
            }
        return None
    
    def _token_to_string(self, token) -> str:
        """Convert token to string, including compounds"""
        if token.dep_ == "compound":
            return ' '.join([t.text for t in token.subtree])
        return token.text
    
    def _fallback_extraction(self, text: str) -> List[Proposition]:
        """Fallback extraction without spaCy"""
        sentences = re.split(r'[.!?]+', text)
        propositions = []
        
        for i, sent in enumerate(sentences):
            sent = sent.strip()
            if len(sent) > 10:  # Skip very short fragments
                # Simple heuristic extraction
                prop = Proposition(
                    id=f"P{i}",
                    text=sent,
                    predicate=self._extract_main_verb(sent),
                    arguments=self._extract_entities(sent),
                    polarity=not ("not" in sent.lower() or "n't" in sent.lower()),
                    modal_strength=0.8,
                    context_position=i
                )
                propositions.append(prop)
        
        return propositions
    
    def _extract_main_verb(self, text: str) -> str:
        """Simple verb extraction"""
        # Very basic verb identification
        words = text.split()
        for word in words:
            if word.endswith(('s', 'ed', 'ing')) and len(word) > 3:
                return word
        return "relates"
    
    def _extract_entities(self, text: str) -> List[str]:
        """Simple entity extraction"""
        # Very basic - find capitalized words and quoted terms
        entities = []
        words = text.split()
        for word in words:
            if word[0].isupper() and word.isalpha():
                entities.append(word)
        return entities[:2]  # Limit to 2 arguments

class ArgumentDetector:
    """Detects logical argument patterns in propositions"""
    
    def __init__(self):
        self.proposition_extractor = PropositionExtractor()
    
    def detect_arguments(self, text: str) -> List[ArgumentInstance]:
        """Detect logical arguments in text"""
        propositions = self.proposition_extractor.extract_propositions(text)
        arguments = []
        
        # Build dependency graph
        prop_graph = self._build_proposition_graph(propositions, text)
        
        # Detect different argument types
        arguments.extend(self._detect_modus_ponens(propositions, text))
        arguments.extend(self._detect_modus_tollens(propositions, text))
        arguments.extend(self._detect_disjunctive_syllogism(propositions, text))
        arguments.extend(self._detect_hypothetical_syllogism(propositions, text))
        arguments.extend(self._detect_contraposition(propositions, text))
        arguments.extend(self._detect_reductio_ad_absurdum(propositions, text))
        arguments.extend(self._detect_abductive_reasoning(propositions, text))
        arguments.extend(self._detect_analogical_reasoning(propositions, text))
        arguments.extend(self._detect_causal_reasoning(propositions, text))
        arguments.extend(self._detect_inductive_generalization(propositions, text))
        
        return arguments
    
    def _build_proposition_graph(self, propositions: List[Proposition], text: str):
        """Build a graph showing relationships between propositions"""
        if not HAS_NETWORKX:
            # Return a simple dictionary representation
            relationships = {}
            for i, prop1 in enumerate(propositions):
                for j, prop2 in enumerate(propositions[i+1:], i+1):
                    start_pos = text.find(prop1.text)
                    end_pos = text.find(prop2.text) + len(prop2.text)
                    
                    if start_pos != -1 and end_pos != -1:
                        between_text = text[start_pos + len(prop1.text):text.find(prop2.text)]
                        relationship = self._identify_relationship(between_text)
                        
                        if relationship:
                            relationships[(prop1.id, prop2.id)] = relationship
            return relationships
        
        # Use NetworkX if available
        G = nx.Graph()
        
        for prop in propositions:
            G.add_node(prop.id, proposition=prop)
        
        # Add edges based on textual proximity and logical indicators
        for i, prop1 in enumerate(propositions):
            for j, prop2 in enumerate(propositions[i+1:], i+1):
                # Check for logical connectors between propositions
                start_pos = text.find(prop1.text)
                end_pos = text.find(prop2.text) + len(prop2.text)
                
                if start_pos != -1 and end_pos != -1:
                    between_text = text[start_pos + len(prop1.text):text.find(prop2.text)]
                    relationship = self._identify_relationship(between_text)
                    
                    if relationship:
                        G.add_edge(prop1.id, prop2.id, relationship=relationship)
        
        return G
    
    def _identify_relationship(self, text: str) -> Optional[str]:
        """Identify logical relationship in connecting text"""
        text = text.lower().strip()
        
        # Implication indicators
        if any(indicator in text for indicator in ["therefore", "thus", "hence", "so", "consequently"]):
            return "implies"
        elif any(indicator in text for indicator in ["if", "when", "given that", "assuming"]):
            return "conditional"
        elif any(indicator in text for indicator in ["but", "however", "although", "while"]):
            return "contrast"
        elif any(indicator in text for indicator in ["and", "also", "furthermore", "moreover"]):
            return "conjunction"
        elif any(indicator in text for indicator in ["or", "either", "alternatively"]):
            return "disjunction"
        elif any(indicator in text for indicator in ["because", "since", "due to", "as"]):
            return "causation"
        
        return None
    
    def _detect_modus_ponens(self, propositions: List[Proposition], text: str) -> List[ArgumentInstance]:
        """Detect modus ponens arguments: If P then Q, P, therefore Q"""
        arguments = []
        
        # Look for conditional statements followed by affirmation of antecedent
        for i, prop in enumerate(propositions):
            # Find conditional statements (if-then patterns)
            if self._is_conditional(prop, text):
                antecedent, consequent = self._extract_conditional_parts(prop, text)
                
                # Look for affirmation of antecedent in subsequent propositions
                for j in range(i+1, min(i+3, len(propositions))):  # Check next few propositions
                    next_prop = propositions[j]
                    
                    if self._propositions_match(antecedent, next_prop):
                        # Look for conclusion (consequent)
                        for k in range(j+1, min(j+3, len(propositions))):
                            conclusion_prop = propositions[k]
                            
                            if self._propositions_match(consequent, conclusion_prop):
                                # Found modus ponens!
                                logical_form = f"({antecedent.predicate}({', '.join(antecedent.arguments)}) → {consequent.predicate}({', '.join(consequent.arguments)})) ∧ {antecedent.predicate}({', '.join(antecedent.arguments)}) ⊢ {consequent.predicate}({', '.join(consequent.arguments)})"
                                
                                arguments.append(ArgumentInstance(
                                    argument_type=ArgumentType.MODUS_PONENS,
                                    premises=[prop, next_prop],
                                    conclusion=conclusion_prop,
                                    validity=True,
                                    confidence=0.8,
                                    text_span=f"{prop.text} ... {next_prop.text} ... {conclusion_prop.text}",
                                    logical_form=logical_form
                                ))
        
        return arguments
    
    def _detect_modus_tollens(self, propositions: List[Proposition], text: str) -> List[ArgumentInstance]:
        """Detect modus tollens arguments: If P then Q, not Q, therefore not P"""
        arguments = []
        
        for i, prop in enumerate(propositions):
            if self._is_conditional(prop, text):
                antecedent, consequent = self._extract_conditional_parts(prop, text)
                
                # Look for negation of consequent
                for j in range(i+1, min(i+3, len(propositions))):
                    next_prop = propositions[j]
                    
                    if (self._propositions_match(consequent, next_prop) and 
                        next_prop.polarity != consequent.polarity):
                        
                        # Look for conclusion (negation of antecedent)
                        for k in range(j+1, min(j+3, len(propositions))):
                            conclusion_prop = propositions[k]
                            
                            if (self._propositions_match(antecedent, conclusion_prop) and
                                conclusion_prop.polarity != antecedent.polarity):
                                
                                logical_form = f"({antecedent.predicate}({', '.join(antecedent.arguments)}) → {consequent.predicate}({', '.join(consequent.arguments)})) ∧ ¬{consequent.predicate}({', '.join(consequent.arguments)}) ⊢ ¬{antecedent.predicate}({', '.join(antecedent.arguments)})"
                                
                                arguments.append(ArgumentInstance(
                                    argument_type=ArgumentType.MODUS_TOLLENS,
                                    premises=[prop, next_prop],
                                    conclusion=conclusion_prop,
                                    validity=True,
                                    confidence=0.8,
                                    text_span=f"{prop.text} ... {next_prop.text} ... {conclusion_prop.text}",
                                    logical_form=logical_form
                                ))
        
        return arguments
    
    def _detect_disjunctive_syllogism(self, propositions: List[Proposition], text: str) -> List[ArgumentInstance]:
        """Detect disjunctive syllogism: P or Q, not P, therefore Q"""
        arguments = []
        
        for i, prop in enumerate(propositions):
            if self._is_disjunctive(prop, text):
                option1, option2 = self._extract_disjunctive_parts(prop, text)
                
                # Look for negation of one option
                for j in range(i+1, min(i+3, len(propositions))):
                    next_prop = propositions[j]
                    
                    negated_option = None
                    remaining_option = None
                    
                    if (self._propositions_match(option1, next_prop) and 
                        next_prop.polarity != option1.polarity):
                        negated_option = option1
                        remaining_option = option2
                    elif (self._propositions_match(option2, next_prop) and 
                          next_prop.polarity != option2.polarity):
                        negated_option = option2
                        remaining_option = option1
                    
                    if negated_option and remaining_option:
                        # Look for conclusion (remaining option)
                        for k in range(j+1, min(j+3, len(propositions))):
                            conclusion_prop = propositions[k]
                            
                            if self._propositions_match(remaining_option, conclusion_prop):
                                logical_form = f"({option1.predicate}({', '.join(option1.arguments)}) ∨ {option2.predicate}({', '.join(option2.arguments)})) ∧ ¬{negated_option.predicate}({', '.join(negated_option.arguments)}) ⊢ {remaining_option.predicate}({', '.join(remaining_option.arguments)})"
                                
                                arguments.append(ArgumentInstance(
                                    argument_type=ArgumentType.DISJUNCTIVE_SYLLOGISM,
                                    premises=[prop, next_prop],
                                    conclusion=conclusion_prop,
                                    validity=True,
                                    confidence=0.7,
                                    text_span=f"{prop.text} ... {next_prop.text} ... {conclusion_prop.text}",
                                    logical_form=logical_form
                                ))
        
        return arguments
    
    def _detect_hypothetical_syllogism(self, propositions: List[Proposition], text: str) -> List[ArgumentInstance]:
        """Detect hypothetical syllogism: If P then Q, if Q then R, therefore if P then R"""
        arguments = []
        
        conditionals = [(i, prop) for i, prop in enumerate(propositions) if self._is_conditional(prop, text)]
        
        for i, (idx1, prop1) in enumerate(conditionals):
            for idx2, prop2 in conditionals[i+1:]:
                ant1, cons1 = self._extract_conditional_parts(prop1, text)
                ant2, cons2 = self._extract_conditional_parts(prop2, text)
                
                # Check if consequent of first matches antecedent of second
                if self._propositions_match(cons1, ant2):
                    # Look for conclusion connecting ant1 to cons2
                    for k, prop3 in enumerate(propositions):
                        if k > max(idx1, idx2):  # After both premises
                            if (self._is_conditional(prop3, text)):
                                ant3, cons3 = self._extract_conditional_parts(prop3, text)
                                
                                if (self._propositions_match(ant1, ant3) and 
                                    self._propositions_match(cons2, cons3)):
                                    
                                    logical_form = f"({ant1.predicate}({', '.join(ant1.arguments)}) → {cons1.predicate}({', '.join(cons1.arguments)})) ∧ ({ant2.predicate}({', '.join(ant2.arguments)}) → {cons2.predicate}({', '.join(cons2.arguments)})) ⊢ ({ant1.predicate}({', '.join(ant1.arguments)}) → {cons2.predicate}({', '.join(cons2.arguments)}))"
                                    
                                    arguments.append(ArgumentInstance(
                                        argument_type=ArgumentType.HYPOTHETICAL_SYLLOGISM,
                                        premises=[prop1, prop2],
                                        conclusion=prop3,
                                        validity=True,
                                        confidence=0.7,
                                        text_span=f"{prop1.text} ... {prop2.text} ... {prop3.text}",
                                        logical_form=logical_form
                                    ))
        
        return arguments
    
    def _detect_contraposition(self, propositions: List[Proposition], text: str) -> List[ArgumentInstance]:
        """Detect contraposition: P→Q is equivalent to ¬Q→¬P"""
        arguments = []
        
        for i, prop in enumerate(propositions):
            if self._is_conditional(prop, text):
                antecedent, consequent = self._extract_conditional_parts(prop, text)
                
                # Look for contrapositive form
                for j in range(len(propositions)):
                    if i != j and self._is_conditional(propositions[j], text):
                        ant2, cons2 = self._extract_conditional_parts(propositions[j], text)
                        
                        # Check if it's contrapositive: ¬Q→¬P
                        if (self._propositions_match(consequent, ant2) and 
                            ant2.polarity != consequent.polarity and
                            self._propositions_match(antecedent, cons2) and
                            cons2.polarity != antecedent.polarity):
                            
                            logical_form = f"({antecedent.predicate}({', '.join(antecedent.arguments)}) → {consequent.predicate}({', '.join(consequent.arguments)})) ≡ (¬{consequent.predicate}({', '.join(consequent.arguments)}) → ¬{antecedent.predicate}({', '.join(antecedent.arguments)}))"
                            
                            arguments.append(ArgumentInstance(
                                argument_type=ArgumentType.CONTRAPOSITION,
                                premises=[prop],
                                conclusion=propositions[j],
                                validity=True,
                                confidence=0.8,
                                text_span=f"{prop.text} ... {propositions[j].text}",
                                logical_form=logical_form
                            ))
        
        return arguments
    
    def _detect_reductio_ad_absurdum(self, propositions: List[Proposition], text: str) -> List[ArgumentInstance]:
        """Detect reductio ad absurdum: Assume ¬P, derive contradiction, conclude P"""
        arguments = []
        
        # Look for assumptions followed by contradictions
        assumption_indicators = ["assume", "suppose", "let's say", "if we consider"]
        contradiction_indicators = ["contradiction", "impossible", "can't be", "doesn't make sense"]
        
        for i, prop in enumerate(propositions):
            if any(indicator in prop.text.lower() for indicator in assumption_indicators):
                # Look for contradictions in subsequent propositions
                for j in range(i+1, min(i+5, len(propositions))):
                    next_prop = propositions[j]
                    
                    if any(indicator in next_prop.text.lower() for indicator in contradiction_indicators):
                        # Look for conclusion (opposite of assumption)
                        for k in range(j+1, min(j+3, len(propositions))):
                            conclusion_prop = propositions[k]
                            
                            # Simplified detection - if conclusion seems to negate the assumption
                            if conclusion_prop.polarity != prop.polarity:
                                logical_form = f"Assume ¬P, derive ⊥, therefore P"
                                
                                arguments.append(ArgumentInstance(
                                    argument_type=ArgumentType.REDUCTIO_AD_ABSURDUM,
                                    premises=[prop, next_prop],
                                    conclusion=conclusion_prop,
                                    validity=True,
                                    confidence=0.6,
                                    text_span=f"{prop.text} ... {next_prop.text} ... {conclusion_prop.text}",
                                    logical_form=logical_form
                                ))
        
        return arguments
    
    def _detect_abductive_reasoning(self, propositions: List[Proposition], text: str) -> List[ArgumentInstance]:
        """Detect abductive reasoning: Best explanation inference"""
        arguments = []
        
        explanation_indicators = ["best explanation", "most likely", "probably because", "explains why"]
        
        for i, prop in enumerate(propositions):
            if any(indicator in prop.text.lower() for indicator in explanation_indicators):
                # This proposition is likely a conclusion of abductive reasoning
                # Look for observations that led to this explanation
                premises = []
                for j in range(max(0, i-3), i):
                    premises.append(propositions[j])
                
                if premises:
                    logical_form = f"Best explanation for observations: {prop.predicate}({', '.join(prop.arguments)})"
                    
                    arguments.append(ArgumentInstance(
                        argument_type=ArgumentType.ABDUCTIVE,
                        premises=premises,
                        conclusion=prop,
                        validity=False,  # Abductive reasoning is not deductively valid
                        confidence=0.6,
                        text_span=f"... {prop.text}",
                        logical_form=logical_form
                    ))
        
        return arguments
    
    def _detect_analogical_reasoning(self, propositions: List[Proposition], text: str) -> List[ArgumentInstance]:
        """Detect analogical reasoning"""
        arguments = []
        
        analogy_indicators = ["like", "similar to", "analogous", "just as", "comparable"]
        
        for i, prop in enumerate(propositions):
            if any(indicator in prop.text.lower() for indicator in analogy_indicators):
                # Look for the conclusion drawn from the analogy
                for j in range(i+1, min(i+3, len(propositions))):
                    conclusion_prop = propositions[j]
                    
                    logical_form = f"Analogical: {prop.predicate} ≈ {conclusion_prop.predicate}"
                    
                    arguments.append(ArgumentInstance(
                        argument_type=ArgumentType.ANALOGICAL,
                        premises=[prop],
                        conclusion=conclusion_prop,
                        validity=False,  # Analogical reasoning is not deductively valid
                        confidence=0.5,
                        text_span=f"{prop.text} ... {conclusion_prop.text}",
                        logical_form=logical_form
                    ))
        
        return arguments
    
    def _detect_causal_reasoning(self, propositions: List[Proposition], text: str) -> List[ArgumentInstance]:
        """Detect causal reasoning patterns"""
        arguments = []
        
        causal_indicators = ["causes", "leads to", "results in", "because", "due to"]
        
        for i, prop in enumerate(propositions):
            if any(indicator in prop.text.lower() for indicator in causal_indicators):
                # Try to identify cause and effect
                for j in range(max(0, i-2), min(i+3, len(propositions))):
                    if i != j:
                        other_prop = propositions[j]
                        
                        if j < i:  # Previous proposition might be cause
                            cause, effect = other_prop, prop
                        else:  # Later proposition might be effect
                            cause, effect = prop, other_prop
                        
                        logical_form = f"Causal: {cause.predicate}({', '.join(cause.arguments)}) → {effect.predicate}({', '.join(effect.arguments)})"
                        
                        arguments.append(ArgumentInstance(
                            argument_type=ArgumentType.CAUSAL,
                            premises=[cause],
                            conclusion=effect,
                            validity=False,  # Causal reasoning can be fallible
                            confidence=0.7,
                            text_span=f"{cause.text} ... {effect.text}",
                            logical_form=logical_form
                        ))
                        break  # Only create one causal link per proposition
        
        return arguments
    
    def _detect_inductive_generalization(self, propositions: List[Proposition], text: str) -> List[ArgumentInstance]:
        """Detect inductive generalization patterns"""
        arguments = []
        
        generalization_indicators = ["all", "most", "generally", "typically", "usually", "in general"]
        
        for i, prop in enumerate(propositions):
            if any(indicator in prop.text.lower() for indicator in generalization_indicators):
                # Look for specific instances that support this generalization
                premises = []
                for j in range(max(0, i-3), i):
                    prev_prop = propositions[j]
                    # If previous propositions share similar predicates, they might be supporting instances
                    if prev_prop.predicate == prop.predicate:
                        premises.append(prev_prop)
                
                if premises:
                    logical_form = f"∀x: {prop.predicate}(x) (generalized from specific instances)"
                    
                    arguments.append(ArgumentInstance(
                        argument_type=ArgumentType.INDUCTIVE_GENERALIZATION,
                        premises=premises,
                        conclusion=prop,
                        validity=False,  # Inductive reasoning is not deductively valid
                        confidence=0.6,
                        text_span=f"... {prop.text}",
                        logical_form=logical_form
                    ))
        
        return arguments
    
    def _is_conditional(self, prop: Proposition, text: str) -> bool:
        """Check if proposition expresses a conditional statement"""
        conditional_markers = ["if", "when", "whenever", "provided", "given that", "assuming"]
        return any(marker in prop.text.lower() for marker in conditional_markers)
    
    def _is_disjunctive(self, prop: Proposition, text: str) -> bool:
        """Check if proposition expresses a disjunction"""
        disjunction_markers = ["or", "either", "alternatively"]
        return any(marker in prop.text.lower() for marker in disjunction_markers)
    
    def _extract_conditional_parts(self, prop: Proposition, text: str) -> Tuple[Proposition, Proposition]:
        """Extract antecedent and consequent from conditional statement"""
        # Simplified extraction - would need more sophisticated parsing
        # For now, create dummy propositions
        antecedent = Proposition(
            id=f"{prop.id}_ant",
            text="antecedent",
            predicate=f"{prop.predicate}_condition",
            arguments=prop.arguments,
            polarity=True,
            modal_strength=prop.modal_strength,
            context_position=prop.context_position
        )
        
        consequent = Proposition(
            id=f"{prop.id}_cons",
            text="consequent", 
            predicate=f"{prop.predicate}_result",
            arguments=prop.arguments,
            polarity=True,
            modal_strength=prop.modal_strength,
            context_position=prop.context_position
        )
        
        return antecedent, consequent
    
    def _extract_disjunctive_parts(self, prop: Proposition, text: str) -> Tuple[Proposition, Proposition]:
        """Extract parts from disjunctive statement"""
        # Simplified - would need better parsing
        option1 = Proposition(
            id=f"{prop.id}_opt1",
            text="option1",
            predicate=f"{prop.predicate}_option1",
            arguments=prop.arguments,
            polarity=True,
            modal_strength=prop.modal_strength,
            context_position=prop.context_position
        )
        
        option2 = Proposition(
            id=f"{prop.id}_opt2",
            text="option2",
            predicate=f"{prop.predicate}_option2", 
            arguments=prop.arguments,
            polarity=True,
            modal_strength=prop.modal_strength,
            context_position=prop.context_position
        )
        
        return option1, option2
    
    def _propositions_match(self, prop1: Proposition, prop2: Proposition) -> bool:
        """Check if two propositions refer to the same logical content"""
        # Simplified matching - could be much more sophisticated
        return (prop1.predicate == prop2.predicate and 
                prop1.arguments == prop2.arguments)

class FormalLogicAnalyzer:
    """Main analyzer for formal logic structures in reasoning"""
    
    def __init__(self):
        self.argument_detector = ArgumentDetector()
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze logical structure of a single stream file"""
        stream_content = self._extract_stream_content(file_path)
        if not stream_content:
            return {}
        
        # Detect logical arguments
        arguments = self.argument_detector.detect_arguments(stream_content)
        
        # Analyze argument structure
        analysis = {
            'file_path': file_path,
            'total_arguments': len(arguments),
            'argument_types': self._categorize_arguments(arguments),
            'deductive_arguments': [arg for arg in arguments if arg.validity],
            'inductive_arguments': [arg for arg in arguments if not arg.validity],
            'logical_complexity': self._calculate_logical_complexity(arguments),
            'argument_chains': self._find_argument_chains(arguments),
            'fallacies': self._detect_fallacies(arguments),
            'formal_representations': [arg.logical_form for arg in arguments]
        }
        
        return analysis
    
    def _extract_stream_content(self, file_path: str) -> str:
        """Extract stream of thought content from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find the stream of thought section
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
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""
    
    def _categorize_arguments(self, arguments: List[ArgumentInstance]) -> Dict[str, int]:
        """Categorize arguments by type"""
        categories = {}
        for arg in arguments:
            arg_type = arg.argument_type.value
            categories[arg_type] = categories.get(arg_type, 0) + 1
        return categories
    
    def _calculate_logical_complexity(self, arguments: List[ArgumentInstance]) -> Dict[str, float]:
        """Calculate various complexity metrics"""
        if not arguments:
            return {'total_complexity': 0.0, 'avg_confidence': 0.0, 'deductive_ratio': 0.0}
        
        total_complexity = len(arguments)
        avg_confidence = sum(arg.confidence for arg in arguments) / len(arguments)
        deductive_count = sum(1 for arg in arguments if arg.validity)
        deductive_ratio = deductive_count / len(arguments)
        
        return {
            'total_complexity': total_complexity,
            'avg_confidence': avg_confidence,
            'deductive_ratio': deductive_ratio,
            'unique_argument_types': len(set(arg.argument_type for arg in arguments))
        }
    
    def _find_argument_chains(self, arguments: List[ArgumentInstance]) -> List[Dict]:
        """Find chains of connected arguments"""
        chains = []
        
        # Simple chaining based on proposition overlap
        for i, arg1 in enumerate(arguments):
            for j, arg2 in enumerate(arguments[i+1:], i+1):
                # Check if conclusion of arg1 appears in premises of arg2
                if any(self._propositions_overlap(arg1.conclusion, premise) 
                       for premise in arg2.premises):
                    chains.append({
                        'chain': [arg1.argument_type.value, arg2.argument_type.value],
                        'confidence': min(arg1.confidence, arg2.confidence)
                    })
        
        return chains
    
    def _propositions_overlap(self, prop1: Proposition, prop2: Proposition) -> bool:
        """Check if propositions have semantic overlap"""
        return prop1.predicate == prop2.predicate and prop1.arguments == prop2.arguments
    
    def _detect_fallacies(self, arguments: List[ArgumentInstance]) -> List[Dict]:
        """Detect logical fallacies"""
        fallacies = []
        
        # Check for potential fallacies
        for arg in arguments:
            if arg.confidence < 0.3:
                fallacies.append({
                    'type': 'weak_argument',
                    'description': f'Argument of type {arg.argument_type.value} has low confidence',
                    'argument': arg.logical_form
                })
            
            # Check for affirming the consequent (invalid modus ponens)
            if (arg.argument_type == ArgumentType.MODUS_PONENS and 
                not arg.validity):
                fallacies.append({
                    'type': 'affirming_consequent',
                    'description': 'Invalid conditional reasoning',
                    'argument': arg.logical_form
                })
        
        return fallacies

def main():
    """Main function to run formal logic analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze formal logic structure in Stream of Thought files")
    parser.add_argument("directory", help="Directory containing *_STREAM_ANALYSIS.txt files")
    parser.add_argument("--output", "-o", default="formal_logic_analysis.json",
                       help="Output file for analysis results")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.directory):
        print(f"Error: Directory {args.directory} does not exist")
        return
    
    analyzer = FormalLogicAnalyzer()
    results = {}
    
    # Find all stream analysis files
    import glob
    pattern = os.path.join(args.directory, "*_STREAM_ANALYSIS.txt")
    files = glob.glob(pattern)
    
    print(f"Found {len(files)} stream analysis files")
    
    for file_path in files:
        filename = os.path.basename(file_path)
        print(f"Analyzing {filename}...")
        
        analysis = analyzer.analyze_file(file_path)
        results[filename] = analysis
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nFormal logic analysis saved to {args.output}")
    
    # Print summary
    if results:
        total_arguments = sum(result.get('total_arguments', 0) for result in results.values())
        print(f"\nSummary:")
        print(f"Total logical arguments detected: {total_arguments}")
        
        # Aggregate argument types
        all_types = {}
        for result in results.values():
            for arg_type, count in result.get('argument_types', {}).items():
                all_types[arg_type] = all_types.get(arg_type, 0) + count
        
        print(f"Argument types found:")
        for arg_type, count in sorted(all_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {arg_type.replace('_', ' ').title()}: {count}")

if __name__ == "__main__":
    main()