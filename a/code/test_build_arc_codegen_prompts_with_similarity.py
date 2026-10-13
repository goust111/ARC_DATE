from __future__ import annotations

from importlib.machinery import SourceFileLoader
import inspect
from pathlib import Path
import unittest


SCRIPT_DIR = Path(__file__).resolve().parent
BUILDER = SourceFileLoader(
    "central_arc_codegen_similarity_builder_under_test",
    str(SCRIPT_DIR / "build_arc_codegen_prompts_with_similarity.py"),
).load_module()


class GenerationSpecTests(unittest.TestCase):
    def test_generation_specs_load_and_embed(self):
        specs = BUILDER.load_generation_specs(
            Path("/home/lixinyu/snap/new/spce_dis")
        )
        self.assertEqual(len(specs), 980)
        self.assertTrue(specs["cc9053aa"].path.endswith("cc9053aa_spec.yaml"))
        self.assertIn("generation_strategy:", specs["cc9053aa"].content)

        ref = BUILDER.ReferencePuzzle(
            puzzle_id="cc9053aa",
            source="train",
            code_path="/tmp/cc9053aa.py",
            docstring="reference",
            annotation_path=None,
            annotation=None,
            annotation_type=None,
            tags=frozenset(),
            tokens=frozenset(),
            code_features=frozenset(),
        )
        item = BUILDER.ScoredReference(
            ref=ref,
            score=1.0,
            mechanism_score=1.0,
            lexical_score=0.0,
            code_feature_score=0.0,
            overlap_tags=(),
        )
        block = BUILDER.build_reference_block(
            item,
            1,
            include_code_excerpt=False,
            max_code_chars=100,
            generation_specs=specs,
        )
        self.assertIn("<generation_spec>", block)
        self.assertIn("cc9053aa_spec.yaml", block)
        self.assertIn("```yaml", block)

    def test_generation_specs_do_not_participate_in_similarity_selection(self):
        parameters = inspect.signature(BUILDER.select_references).parameters
        self.assertNotIn("generation_specs", parameters)


if __name__ == "__main__":
    unittest.main()
