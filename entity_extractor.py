import asyncio
from langdetect import DetectorFactory, detect_langs
from cat import log
import hashlib
import re
from typing import List, Dict
from spacy import load as spacy_load
from spacy.util import is_package as spacy_is_package
from spacy.cli.download import download as spacy_download
from spacy.language import Language
from spacy.tokens import Doc

from .constants import SPACY_TO_ENTITY_TYPE, TECHNOLOGY_PATTERNS, MAX_CO_OCCURRENCE_ENTITIES, VERB_TO_RELATION_TYPE
from .models import ExtractedEntity, ExtractedRelation, EntityType, DocumentWithEntities


class EntityExtractor:
    """
    Extracts entities and relations from text using spaCy.
    Supports multilingual models and extension with custom rules.
    """
    def __init__(self, models: Dict[str, str], extra_technology_patterns: List[str] = None):
        """
        Initializes the spaCy model.

        Args:
            models: A dictionary mapping language codes (e.g. "en") to spaCy model names (e.g. "en_core_web_sm").
                The specified models will be loaded on demand when processing documents in the corresponding language.
            extra_technology_patterns: additional regex patterns appended to
                TECHNOLOGY_PATTERNS. Useful for non-English tech terms or
                domain-specific keywords not covered by the built-in list.
        """
        self._models = models
        self._models["default"] = "en_core_web_sm"
        self._models_loaded = {k: False for k in self._models.keys()}

        self._nlps: Dict[str, Language] = {}
        self._initialized = False

        # Build the instance-level pattern list so it can be extended per-instance
        self._technology_patterns = list(TECHNOLOGY_PATTERNS)
        if extra_technology_patterns:
            self._technology_patterns.extend(extra_technology_patterns)

    @staticmethod
    def _download_spacy_model(model_name: str):
        if not spacy_is_package(model_name):
            spacy_download(model_name)

    @staticmethod
    def _detect_language(text: str) -> str | None:
        DetectorFactory.seed = 0
        text = text.strip()
        if len(text) < 5:
            return None
        try:
            langs = detect_langs(text)
            langs = [l.lang for l in langs if l.prob > 0.8]
            if len(langs) == 0:
                return None
            return langs[0]
        except Exception:
            return None

    def _load_model(self, lang: str, model_name: str):
        if self._models_loaded[lang]:
            return
        try:
            self._download_spacy_model(model_name)
            self._nlps[lang] = spacy_load(model_name)
            self._models_loaded[lang] = True

            log.info(f"Loaded spaCy model '{model_name}' for language '{lang}'")
        except Exception as e:
            log.error(f"Failed to load spaCy model '{model_name}' for language '{lang}': {e}")

    async def ensure_downloaded(self):
        """Downloads the spaCy model (asynchronously)."""
        await asyncio.gather(*[
            asyncio.to_thread(self._download_spacy_model, model_name)
            for lang, model_name in self._models.items()
        ])

    async def ensure_initialized(self):
        """Loads the spaCy model (asynchronously)."""
        if self._initialized:
            return

        # parallel loads each self._models into the self.nlps in separate threads because this is blocking and can take
        # several seconds
        await asyncio.gather(*[
            asyncio.to_thread(self._load_model, lang, model_name)
            for lang, model_name in self._models.items()
        ])

        self._initialized = True

    async def extract_doc(self, text: str) -> Doc:
        if not self._initialized:
            await self.ensure_initialized()

        lang = self._detect_language(text)
        nlp = self._nlps.get(lang, self._nlps["default"]) if lang else self._nlps["default"]
        return nlp(text)

    async def extract(self, text: str, document_id: str, metadata: Dict = None) -> DocumentWithEntities:
        """
        Extracts entities and relations from text.
        """
        if not self._initialized:
            await self.ensure_initialized()

        # Process text with spaCy
        lang = self._detect_language(text)
        nlp = self._nlps.get(lang, self._nlps["default"]) if lang else self._nlps["default"]
        doc: Doc = await asyncio.to_thread(nlp, text)
        
        # Extract entities
        extracted_entities = self.extract_entities(doc)
        
        # Add technology entities with regex (captures things spaCy might miss)
        extracted_entities.extend(self.extract_technologies_regex(text))
        
        # Deduplicate entities
        extracted_entities = self.deduplicate_entities(extracted_entities)
        
        # Extract relations
        relations = self._extract_relations(doc, extracted_entities, text)
        
        # Add co-occurrence relations only for small entity sets.
        # N*(N-1)/2 pairs are too expensive for large documents.
        if len(extracted_entities) <= MAX_CO_OCCURRENCE_ENTITIES:
            relations.extend([
                ExtractedRelation(
                    source_entity=extracted_entities[i].name,
                    target_entity=extracted_entities[j].name,
                    relation_type="CO_OCCURS_WITH",
                    weight=(extracted_entities[i].confidence + extracted_entities[j].confidence) / 4
                )
                for i in range(len(extracted_entities))
                for j in range(i + 1, len(extracted_entities))
            ])
        
        return DocumentWithEntities(
            document_id=document_id,
            content=text,
            entities=extracted_entities,
            relations=relations,
            metadata=metadata or {}
        )

    @staticmethod
    def extract_entities(doc: Doc) -> List[ExtractedEntity]:
        """Extracts named entities from the spaCy document."""
        entities = []
        for ent in doc.ents:
            entity_type = SPACY_TO_ENTITY_TYPE.get(ent.label_, EntityType.UNKNOWN)
            entities.append(ExtractedEntity(
                name=ent.text.strip(),
                type=entity_type,
                start_char=ent.start_char,
                end_char=ent.end_char,
                confidence=0.9 if entity_type != EntityType.UNKNOWN else 0.5
            ))
        return entities

    def extract_technologies_regex(self, text: str) -> List[ExtractedEntity]:
        """Extracts technology entities using the instance-level regex pattern list.

        Complements spaCy NER for tech terms that statistical models often miss
        (framework names, acronyms, brand names).  The instance-level list can be
        extended at construction time via `extra_technology_patterns`.
        """
        return [
            ExtractedEntity(
                name=match.group(0),
                type=EntityType.TECHNOLOGY,
                start_char=match.start(),
                end_char=match.end(),
                confidence=0.85,
            )
            for pattern in self._technology_patterns
            for match in re.finditer(pattern, text, re.IGNORECASE)
        ]

    @staticmethod
    def deduplicate_entities(entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Deduplicates nearby or identical entities."""
        # Normalize names
        normalized = {}
        for ent in entities:
            name_lower = ent.name.lower().strip()
            
            if name_lower in normalized:
                # Keep the one with higher confidence
                existing = normalized[name_lower]
                if ent.confidence > existing.confidence:
                    normalized[name_lower] = ent
            else:
                normalized[name_lower] = ent
                
        return list(normalized.values())

    @staticmethod
    def _extract_relations(
        doc: Doc,
        entities: List[ExtractedEntity],
        full_text: str
    ) -> List[ExtractedRelation]:
        """
        Extracts relations between entities in two phases:

        1. **Dependency-parse phase** (primary, weight=0.85):
           Uses spaCy's dependency tree to find Subject-Verb-Object triples.
           Passive voice is supported via the "agent" arc
           (e.g. "Python was developed by Guido" → DEVELOPS(Guido, Python)).

        2. **Proximity-fallback phase** (secondary, weight=0.6):
           For pairs not already found by the parser, a 100-char sliding window
           catches implicit relations (enumerations, appositions, etc.) using
           lightweight keyword patterns.
        """
        relations: List[ExtractedRelation] = []

        if len(entities) < 2:
            return relations

        sorted_entities = sorted(entities, key=lambda e: e.start_char)

        # ── helpers ──────────────────────────────────────────────────────────

        def find_entity_for_span(start: int, end: int) -> ExtractedEntity | None:
            """Returns the first entity whose char-span overlaps [start, end)."""
            for ent in sorted_entities:
                if ent.start_char < end and start < ent.end_char:
                    return ent
            return None

        def find_entity_for_token(tok) -> ExtractedEntity | None:
            """
            Tries to resolve a dependency-tree token to one of our entities.

            1. Checks the token's own character span.
            2. If not found, expands to the token's subtree (handles compound
               proper nouns like "New York" or "Apache Kafka").
               The expansion is capped at 5 tokens to avoid matching full clauses.
            """
            ent = find_entity_for_span(tok.idx, tok.idx + len(tok.text))
            if ent:
                return ent
            subtree = list(tok.subtree)
            if 2 <= len(subtree) <= 5:
                span_start = subtree[0].idx
                span_end = subtree[-1].idx + len(subtree[-1].text)
                return find_entity_for_span(span_start, span_end)
            return None

        # ── phase 1: SVO via dependency parser ───────────────────────────────

        seen_pairs: set = set()

        for token in doc:
            if token.pos_ not in ("VERB", "AUX"):
                continue

            # Grammatical subjects (active and passive)
            subjects = [
                child for child in token.children
                if child.dep_ in ("nsubj", "nsubjpass")
            ]

            # Grammatical objects: direct, attributive, prepositional
            objects = [
                child for child in token.children
                if child.dep_ in ("dobj", "attr", "acomp", "oprd")
            ]
            for prep in (c for c in token.children if c.dep_ == "prep"):
                objects.extend(
                    gc for gc in prep.children if gc.dep_ in ("pobj", "pcomp")
                )

            # Passive agent: "X was built by Y" → "by Y" agent arc
            # The pobj of "by" becomes the logical subject.
            for agent in (c for c in token.children if c.dep_ == "agent"):
                subjects.extend(gc for gc in agent.children if gc.dep_ == "pobj")

            rel_type = VERB_TO_RELATION_TYPE.get(token.lemma_, "RELATED_TO")

            for subj_tok in subjects:
                for obj_tok in objects:
                    subj_ent = find_entity_for_token(subj_tok)
                    obj_ent = find_entity_for_token(obj_tok)

                    if not subj_ent or not obj_ent or subj_ent.name == obj_ent.name:
                        continue

                    pair_key = (subj_ent.name, obj_ent.name, rel_type)
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    relations.append(ExtractedRelation(
                        source_entity=subj_ent.name,
                        target_entity=obj_ent.name,
                        relation_type=rel_type,
                        weight=0.85,  # syntactically grounded → higher confidence
                    ))

        # ── phase 2: proximity fallback ───────────────────────────────────────
        # Catches pairs not expressed through explicit verbs
        # (appositions, enumerations, nominal predicates, etc.)

        found_pairs = {(r.source_entity, r.target_entity) for r in relations}

        for i in range(len(sorted_entities)):
            e1 = sorted_entities[i]
            for j in range(i + 1, len(sorted_entities)):
                e2 = sorted_entities[j]

                if e2.start_char - e1.end_char >= 100:
                    break  # beyond proximity window, skip remaining

                if (e1.name, e2.name) in found_pairs:
                    continue  # already captured by the dependency phase

                between = full_text[e1.end_char:e2.start_char].strip().lower()

                rel_type = "RELATED_TO"
                if "is a" in between or "are a" in between:
                    rel_type = "IS_A"
                elif "based on" in between:
                    rel_type = "BASED_ON"

                relations.append(ExtractedRelation(
                    source_entity=e1.name,
                    target_entity=e2.name,
                    relation_type=rel_type,
                    weight=0.6,  # heuristic → lower confidence
                ))

        return relations

    @staticmethod
    def get_entity_hash(name: str, entity_type: EntityType, tenant_id: str) -> str:
        """Generates a unique hash for an entity (for cross-document deduplication)."""
        # Remove extra spaces, convert to lowercase
        normalized = name.lower().strip()
        # Remove leading articles
        normalized = re.sub(r'^(the|a|an)\s+', '', normalized)

        key = f"{tenant_id}:{entity_type.value}:{normalized}"
        return hashlib.md5(key.encode()).hexdigest()
