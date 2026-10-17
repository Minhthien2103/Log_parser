"""Mask-first parsing on top of the real ``drain3`` library.

The paper evaluates DeepParse with "an extended version of Drain3": the
open-source Drain3 implementation, fed with lines whose variable spans were
already replaced by typed placeholders.  :class:`Drain3Engine` is that setup.
It applies the same :class:`~deepparse.drain.masks_application.MaskApplier`
as :class:`~deepparse.drain.drain_engine.DrainEngine`, then hands the masked
line to ``drain3.TemplateMiner`` and renders every placeholder in the mined
template as ``<*>``.

Compared with the in-repo port, Drain3 follows the original algorithm when it
walks the parse tree: ``depth`` counts the root and leaf layers (so ``depth=5``
means three prefix-token layers), and any token that contains a digit is a
wildcard at that stage (``parametrize_numeric_tokens``).  The engine exposes
the same ``parse`` / ``parse_with_ids`` / ``clusters`` surface so the
evaluation runner can switch between the two with one config key.

Drain3's own masking is left empty on purpose: masking is DeepParse's job and
is done before the line reaches the miner.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from deepparse.drain.drain_engine import DrainCluster, _to_wildcard
from deepparse.drain.masks_application import MaskApplier
from deepparse.masks_types import Mask
from deepparse.tokenize import tokenize


@dataclass
class Drain3Engine:
    """Drain3-backed engine with DeepParse mask-first preprocessing.

    Defaults match the Drain3 baseline configuration cited in the paper
    (Section "Baseline Parsers"): ``depth=5``, ``similarity_threshold=0.4``,
    ``max_children=100``.
    """

    depth: int = 5
    similarity_threshold: float = 0.4
    max_children: int = 100
    masks: Sequence[Mask] = field(default_factory=list)

    def __post_init__(self) -> None:
        try:
            from drain3 import TemplateMiner
            from drain3.template_miner_config import TemplateMinerConfig
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Drain3Engine needs the optional 'drain3' extra: pip install -e \".[drain3]\""
            ) from exc

        self.applier = MaskApplier(self.masks)
        config = TemplateMinerConfig()
        config.drain_depth = self.depth
        config.drain_sim_th = self.similarity_threshold
        config.drain_max_children = self.max_children
        config.drain_max_clusters = None
        config.parametrize_numeric_tokens = True
        config.masking_instructions = []  # DeepParse masks run first; Drain3 masks nothing
        config.profiling_enabled = False
        self._miner = TemplateMiner(config=config)

    # ---- public API --------------------------------------------------------
    def add_log(self, line: str) -> int:
        """Feed one line; return its Drain3 cluster id."""
        masked = self.applier.apply(line)
        if not masked.strip():
            masked = ""
        result = self._miner.add_log_message(masked)
        return int(result["cluster_id"])

    def _template_of(self, cluster_id: int) -> str:
        cluster = self._miner.drain.id_to_cluster[cluster_id]
        return " ".join(_to_wildcard(tok) for tok in tokenize(cluster.get_template()))

    def parse_with_ids(self, lines: Iterable[str]) -> list[tuple[int, str]]:
        """Two-pass parse: add every line, then read each cluster's final template."""
        ids = [self.add_log(line) for line in lines]
        templates = {cid: self._template_of(cid) for cid in set(ids)}
        return [(cid, templates[cid]) for cid in ids]

    def parse(self, lines: Iterable[str]) -> list[str]:
        return [template for _, template in self.parse_with_ids(lines)]

    @property
    def clusters(self) -> list[DrainCluster]:
        """Clusters as :class:`DrainCluster` records (template tokens, size), ordered by id."""
        out = []
        for cid, cluster in sorted(self._miner.drain.id_to_cluster.items()):
            tokens = [_to_wildcard(tok) for tok in tokenize(cluster.get_template())]
            out.append(DrainCluster(cluster_id=int(cid), template=tokens, size=int(cluster.size)))
        return out

    @property
    def num_clusters(self) -> int:
        return len(self._miner.drain.id_to_cluster)
