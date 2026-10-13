from __future__ import annotations

from importlib.machinery import SourceFileLoader
import inspect
from pathlib import Path
import unittest


SCRIPT_DIR = Path(__file__).resolve().parent
BUILDER = SourceFileLoader(
    "arc_codegen_similarity_builder_under_test",
    str(SCRIPT_DIR / "build_arc_codegen_prompts_with_similarity.py"),
).load_module()
DES_DIR = Path("/home/lixinyu/snap/a/key_combo/des/fable/1")


class NewPuzzleIsolationTests(unittest.TestCase):
    def puzzle(self, filename: str, index: str):
        puzzles = {
            puzzle.index: puzzle
            for puzzle in BUILDER.parse_new_puzzles(DES_DIR / filename)
        }
        return puzzles[index]

    def test_declared_tags_are_verbatim_and_isolated(self):
        puzzle = self.puzzle("034.md", "5")
        self.assertEqual(
            puzzle.tags,
            frozenset(
                {
                    "operation:寻路, pathfinding",
                    "operation:重涂色, recolor",
                    "object:轨道路径, track_path",
                    "object:端点, endpoint",
                    "object:传送门, portal",
                    "relation:同色配对, same_color_pairing",
                    "condition:最短路径唯一, unique_shortest_path",
                    "condition:诱饵绕行路, detour_decoy",
                }
            ),
        )
        self.assertNotIn("object:孔洞, hole", puzzle.tags)
        self.assertIn("code:bfs", puzzle.expected_code_features)

    def test_explicit_references_only_come_from_current_block(self):
        puzzle = self.puzzle("043.md", "6")
        self.assertEqual(
            puzzle.explicit_reference_ids,
            frozenset({"dd6b8c4b", "ff72ca3e"}),
        )
        self.assertNotIn("1d398264", puzzle.explicit_reference_ids)
        self.assertEqual(len(puzzle.group_reference_ids), 20)

    def test_file_header_group_is_shared_but_not_explicit(self):
        puzzle = self.puzzle("025.md", "6")
        self.assertEqual(puzzle.explicit_reference_ids, frozenset())
        self.assertEqual(len(puzzle.group_reference_ids), 20)
        self.assertIn(
            "condition:像素数守恒, pixel_count_conservation",
            puzzle.tags,
        )

    def test_keyword_extraction_is_only_a_missing_tags_fallback(self):
        description = """## 1. fallback\n\n迷宫中寻找唯一最短路径并重新着色。"""
        puzzle = BUILDER.make_new_puzzle("1", description, full_text=description)
        self.assertIn("operation:bfs_distance", puzzle.tags)

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
