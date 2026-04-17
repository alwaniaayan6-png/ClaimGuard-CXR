"""Dataset loaders and download helpers for ClaimGuard-Bench-Grounded.

Every loader returns a uniform schema:

    {
        'image_id': str,
        'patient_id': str | None,
        'image_path': pathlib.Path,
        'annotations': dict[str, list[Annotation]],  # canonical-finding-keyed
        'report_text': str | None,
        'split': 'train' | 'cal' | 'test',
    }

so that downstream grounding and evaluation code does not need to know
which dataset it is processing.
"""
from .ontology_mapping import OntologyMapper, load_ontology

__all__ = ["OntologyMapper", "load_ontology"]
