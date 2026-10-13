from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import Color
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        GRAY = 5
        PINK = 6
        ORANGE = 7
        TEAL = 8
        MAROON = 9
        NOT_BLACK = [1, 2, 3, 4, 5, 6, 7, 8, 9]


COMBO_ORDER = ((False, False), (False, True), (True, False), (True, True))
TRUTH_TABLES = [
    ("and", (0, 0, 0, 1)),
    ("or", (0, 1, 1, 1)),
    ("xor", (0, 1, 1, 0)),
    ("xnor", (1, 0, 0, 1)),
    ("nand", (1, 1, 1, 0)),
    ("nor", (1, 0, 0, 0)),
    ("left_and_not_right", (0, 0, 1, 0)),
    ("not_left_and_right", (0, 1, 0, 0)),
    ("left_implies_right", (1, 1, 0, 1)),
    ("right_implies_left", (1, 0, 1, 1)),
]
TRUTH_POOL: list[tuple[str, tuple[int, int, int, int]]] = []

SEP_WIDTH_TO_DEMO_WIDTHS = {
    1: (3, 5, 7),
    2: (4, 6),
}
MAX_GENERATION_ATTEMPTS = 400


def _panel_colors(panel: np.ndarray) -> set[int]:
    return set(map(int, np.unique(panel)))


def _truth_dict(truth: tuple[int, int, int, int] | np.ndarray) -> dict[tuple[bool, bool], bool]:
    values = tuple(bool(v) for v in truth)
    return {combo: values[i] for i, combo in enumerate(COMBO_ORDER)}


def _truth_tuple(table: dict[tuple[bool, bool], bool]) -> tuple[int, int, int, int]:
    return tuple(int(table[combo]) for combo in COMBO_ORDER)


def _uses_both_inputs(table: dict[tuple[bool, bool], bool]) -> bool:
    left_matters = (table[(False, False)] != table[(True, False)]) or (table[(False, True)] != table[(True, True)])
    right_matters = (table[(False, False)] != table[(False, True)]) or (table[(True, False)] != table[(True, True)])
    return bool(left_matters and right_matters)


def _apply_truth(left_mask: np.ndarray, right_mask: np.ndarray, table: dict[tuple[bool, bool], bool]) -> np.ndarray:
    out = np.zeros(left_mask.shape, dtype=bool)
    for combo, value in table.items():
        if value:
            out |= (left_mask == combo[0]) & (right_mask == combo[1])
    return out


def _mask_from_panel(panel: np.ndarray, bg: int, active_color: int) -> np.ndarray:
    colors = _panel_colors(panel)
    assert colors.issubset({bg, active_color}), "panel contains an unexpected color"
    return panel == active_color


def _split_band_into_panels(band: np.ndarray, separator_color: int) -> list[np.ndarray]:
    assert band.ndim == 2 and band.shape[0] > 0 and band.shape[1] > 0
    is_separator_col = np.all(band == separator_color, axis=0)
    spans: list[tuple[int, int]] = []
    c = 0
    width = int(band.shape[1])
    while c < width:
        while c < width and is_separator_col[c]:
            c += 1
        if c >= width:
            break
        start = c
        while c < width and not is_separator_col[c]:
            c += 1
        spans.append((start, c))
    assert spans, "expected at least one panel span"
    panels = [np.array(band[:, start:end], dtype=int) for start, end in spans]
    for panel in panels:
        assert separator_color not in panel, "separator color must not appear inside a panel"
    return panels


def _extract_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, _width = map(int, g.shape)

    candidates: list[dict[str, object]] = []
    for r in range(1, height - 1):
        if not np.all(g[r, :] == g[r, 0]):
            continue
        separator_color = int(g[r, 0])
        top = g[:r, :]
        bottom = g[r + 1 :, :]
        if top.size == 0 or bottom.size == 0:
            continue
        try:
            top_panels = _split_band_into_panels(top, separator_color)
            bottom_panels = _split_band_into_panels(bottom, separator_color)
        except AssertionError:
            continue
        if len(top_panels) != 3 or len(bottom_panels) != 2:
            continue
        candidates.append(
            {
                "separator_row": r,
                "separator_color": separator_color,
                "top_panels": top_panels,
                "bottom_panels": bottom_panels,
            }
        )

    assert len(candidates) == 1, "expected a unique demo/main separator row"
    scene = candidates[0]

    p_panel, q_panel, r_panel = scene["top_panels"]
    a_panel, b_panel = scene["bottom_panels"]

    demo_shape = p_panel.shape
    assert q_panel.shape == demo_shape and r_panel.shape == demo_shape, "demo panels must have equal size"
    main_shape = a_panel.shape
    assert b_panel.shape == main_shape, "main panels must have equal size"

    separator_color = int(scene["separator_color"])
    pqab_colors = set()
    for panel in (p_panel, q_panel, a_panel, b_panel):
        pqab_colors |= _panel_colors(panel)
    pqab_colors.discard(separator_color)
    r_colors = _panel_colors(r_panel)
    r_colors.discard(separator_color)

    bg_candidates = pqab_colors & r_colors
    assert len(bg_candidates) == 1, "expected a unique shared background color"
    bg = int(next(iter(bg_candidates)))

    input_candidates = pqab_colors - {bg}
    result_candidates = r_colors - {bg}
    assert len(input_candidates) == 1, "P/Q/A/B must share one active color"
    assert len(result_candidates) == 1, "R must use one result color"

    input_color = int(next(iter(input_candidates)))
    result_color = int(next(iter(result_candidates)))
    assert input_color != result_color != separator_color

    return {
        "bg": bg,
        "separator_color": separator_color,
        "input_color": input_color,
        "result_color": result_color,
        "P": p_panel,
        "Q": q_panel,
        "R": r_panel,
        "A": a_panel,
        "B": b_panel,
    }


def _infer_truth_table(scene: dict[str, object]) -> dict[tuple[bool, bool], bool]:
    p_mask = _mask_from_panel(scene["P"], scene["bg"], scene["input_color"])
    q_mask = _mask_from_panel(scene["Q"], scene["bg"], scene["input_color"])
    r_mask = _mask_from_panel(scene["R"], scene["bg"], scene["result_color"])

    table: dict[tuple[bool, bool], bool] = {}
    for left, right, out in zip(p_mask.ravel(), q_mask.ravel(), r_mask.ravel()):
        combo = (bool(left), bool(right))
        value = bool(out)
        if combo in table:
            assert table[combo] == value, "demo gives conflicting outputs for one input combo"
        else:
            table[combo] = value

    assert len(table) == 4, "demo must cover all four input combinations"
    assert _uses_both_inputs(table), "operator must depend on both demo inputs"
    return table


def main(grid: np.ndarray) -> np.ndarray:
    """Infer the demo-defined pixelwise logic operator and apply it to the lower mask pair.

    Input structure:
    - One separator color divides the grid into an upper demo row and a lower main row.
    - The upper row contains three equal-size panels P, Q, R; the lower row contains two equal-size panels A, B.
    - P, Q, A, and B use one active color on a shared background, while R uses a different active color on that same background.

    Rule:
    - Read P, Q, and R as a complete truth-table demonstration of one binary pixelwise logic operator.
    - Use the four possible per-cell input combinations across P and Q to infer whether the operator outputs true or false for each case.
    - Apply that inferred operator to corresponding cells of A and B, and output a grid with A's size where true cells take R's active color and false cells stay background.

    Key insight:
    - The rule is not fixed ahead of time; each input carries its own full operator specification in the top demo triplet, and the bottom pair must be solved by transferring that learned truth table.
    """
    scene = _extract_scene(grid)
    table = _infer_truth_table(scene)

    a_mask = _mask_from_panel(scene["A"], scene["bg"], scene["input_color"])
    b_mask = _mask_from_panel(scene["B"], scene["bg"], scene["input_color"])
    result_mask = _apply_truth(a_mask, b_mask, table)

    out = np.full(a_mask.shape, int(scene["bg"]), dtype=int)
    out[result_mask] = int(scene["result_color"])
    return out


def _next_truth() -> tuple[str, tuple[int, int, int, int]]:
    global TRUTH_POOL
    if not TRUTH_POOL:
        TRUTH_POOL = TRUTH_TABLES.copy()
        random.shuffle(TRUTH_POOL)
    return TRUTH_POOL.pop()


def _compatible_widths(separator_width: int) -> tuple[int, int]:
    demo_width = int(random.choice(SEP_WIDTH_TO_DEMO_WIDTHS[separator_width]))
    main_width = (separator_width + 3 * demo_width) // 2
    return demo_width, main_width


def _compose_band(panels: list[np.ndarray], separator_color: int, separator_width: int) -> np.ndarray:
    height = int(panels[0].shape[0])
    assert all(panel.shape[0] == height for panel in panels)
    total_width = sum(int(panel.shape[1]) for panel in panels) + separator_width * (len(panels) + 1)
    band = np.full((height, total_width), separator_color, dtype=int)
    c = separator_width
    for panel in panels:
        width = int(panel.shape[1])
        band[:, c : c + width] = panel
        c += width + separator_width
    return band


def _combo_grid_to_masks(combo_grid: np.ndarray, truth: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_mask = combo_grid >= 2
    q_mask = (combo_grid % 2) == 1
    truth_arr = np.asarray(truth, dtype=bool)
    r_mask = truth_arr[combo_grid]
    return p_mask, q_mask, r_mask


def _masks_to_panel(mask: np.ndarray, active_color: int, bg: int) -> np.ndarray:
    panel = np.full(mask.shape, bg, dtype=int)
    panel[mask] = active_color
    return panel


def _combo_counts(combo_grid: np.ndarray) -> dict[int, int]:
    values, counts = np.unique(combo_grid, return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}


def _sample_combo_grid(height: int, width: int, truth: tuple[int, int, int, int], min_each: int) -> np.ndarray | None:
    area = height * width
    if 4 * min_each > area:
        return None

    truth_arr = np.asarray(truth, dtype=bool)
    for _ in range(200):
        combo_values = [0, 1, 2, 3] * min_each
        combo_values.extend(random.randrange(4) for _ in range(area - len(combo_values)))
        random.shuffle(combo_values)
        combo_grid = np.asarray(combo_values, dtype=int).reshape(height, width)

        p_mask = combo_grid >= 2
        q_mask = (combo_grid % 2) == 1
        r_mask = truth_arr[combo_grid]

        p_density = float(p_mask.mean())
        q_density = float(q_mask.mean())
        r_density = float(r_mask.mean())
        if not (0.20 <= p_density <= 0.80 and 0.20 <= q_density <= 0.80):
            continue
        if not (0.10 <= r_density <= 0.90):
            continue
        if min_each >= 2 and not (2 <= int(r_mask.sum()) <= area - 2):
            continue
        return combo_grid
    return None


def _validate_generated_scene(grid: np.ndarray, truth: tuple[int, int, int, int]) -> bool:
    try:
        scene = _extract_scene(grid)
        table = _infer_truth_table(scene)
        if _truth_tuple(table) != tuple(int(v) for v in truth):
            return False

        demo_combo_grid = (
            2 * _mask_from_panel(scene["P"], scene["bg"], scene["input_color"]).astype(int)
            + _mask_from_panel(scene["Q"], scene["bg"], scene["input_color"]).astype(int)
        )
        main_combo_grid = (
            2 * _mask_from_panel(scene["A"], scene["bg"], scene["input_color"]).astype(int)
            + _mask_from_panel(scene["B"], scene["bg"], scene["input_color"]).astype(int)
        )
        demo_counts = _combo_counts(demo_combo_grid)
        main_counts = _combo_counts(main_combo_grid)
        if any(demo_counts.get(combo, 0) < 1 for combo in range(4)):
            return False
        if any(main_counts.get(combo, 0) < 2 for combo in range(4)):
            return False

        expected_output = _masks_to_panel(_apply_truth(main_combo_grid >= 2, (main_combo_grid % 2) == 1, table), scene["result_color"], scene["bg"])
        actual_output = main(grid)
        if not np.array_equal(actual_output, expected_output):
            return False

        a_mask = scene["A"] == scene["input_color"]
        b_mask = scene["B"] == scene["input_color"]
        if np.array_equal(actual_output == scene["result_color"], a_mask):
            return False
        if np.array_equal(actual_output == scene["result_color"], b_mask):
            return False
        return True
    except Exception:
        return False


def _build_case() -> np.ndarray | None:
    separator_width = int(random.choice((1, 2)))
    demo_width, main_width = _compatible_widths(separator_width)
    demo_height = int(random.randint(3, 6))
    main_height = int(random.randint(demo_height + 1, min(12, demo_height + 6)))

    total_width = 3 * demo_width + 4 * separator_width
    total_height = demo_height + 1 + main_height
    if total_width > 30 or total_height > 30:
        return None

    bg = Color.BLACK
    separator_color, input_color, result_color = random.sample(Color.NOT_BLACK, 3)
    _name, truth = _next_truth()

    demo_combo_grid = _sample_combo_grid(demo_height, demo_width, truth, min_each=1)
    main_combo_grid = _sample_combo_grid(main_height, main_width, truth, min_each=2)
    if demo_combo_grid is None or main_combo_grid is None:
        return None

    p_mask, q_mask, r_mask = _combo_grid_to_masks(demo_combo_grid, truth)
    a_mask, b_mask, _unused = _combo_grid_to_masks(main_combo_grid, truth)

    p_panel = _masks_to_panel(p_mask, input_color, bg)
    q_panel = _masks_to_panel(q_mask, input_color, bg)
    r_panel = _masks_to_panel(r_mask, result_color, bg)
    a_panel = _masks_to_panel(a_mask, input_color, bg)
    b_panel = _masks_to_panel(b_mask, input_color, bg)

    top_band = _compose_band([p_panel, q_panel, r_panel], separator_color, separator_width)
    bottom_band = _compose_band([a_panel, b_panel], separator_color, separator_width)
    if top_band.shape[1] != bottom_band.shape[1]:
        return None

    separator_row = np.full((1, top_band.shape[1]), separator_color, dtype=int)
    grid = np.vstack([top_band, separator_row, bottom_band])
    return grid if _validate_generated_scene(grid, truth) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid demo-defined logic operator puzzle")
