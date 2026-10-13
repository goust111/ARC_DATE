from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from arc_utils import Color, bbox_h, bbox_w, crop_bbox, find_components, get_bg


_OBJ_SIZE = 4
_INNER = slice(1, 3)
_COLOR_POOL = tuple(range(1, 10))
_GEN_MAX_ATTEMPTS = 400
_HEIGHT_RANGE = (23, 31)
_SEQ_COUNT_RANGE = (2, 5)
_SEQ_GAP_RANGE = (2, 4)
_MARGIN_RANGE = (0, 3)
_OBJECT_COUNT_OPTIONS = (3, 4)


def _frame_sprite(color: int) -> np.ndarray:
    sprite = np.full((_OBJ_SIZE, _OBJ_SIZE), int(color), dtype=int)
    sprite[_INNER, _INNER] = Color.BLACK
    return sprite


def _solid_sprite(color: int) -> np.ndarray:
    return np.full((_OBJ_SIZE, _OBJ_SIZE), int(color), dtype=int)


def _boxes_overlap(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> bool:
    return not (
        int(box_a[2]) < int(box_b[0])
        or int(box_b[2]) < int(box_a[0])
        or int(box_a[3]) < int(box_b[1])
        or int(box_b[3]) < int(box_a[1])
    )


def _is_frame(sub: np.ndarray, color: int, bg: int) -> bool:
    return (
        sub.shape == (_OBJ_SIZE, _OBJ_SIZE)
        and bool(np.all(sub[0, :] == int(color)))
        and bool(np.all(sub[-1, :] == int(color)))
        and bool(np.all(sub[:, 0] == int(color)))
        and bool(np.all(sub[:, -1] == int(color)))
        and bool(np.all(sub[_INNER, _INNER] == int(bg)))
    )


def _is_solid(sub: np.ndarray, color: int) -> bool:
    return sub.shape == (_OBJ_SIZE, _OBJ_SIZE) and bool(np.all(sub == int(color)))


def _extract_sequences(grid: np.ndarray) -> list[list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = int(get_bg(g))
    assert bg == int(Color.BLACK), "expected a black background"

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    sequences: dict[tuple[int, int], list[dict[str, object]]] = {}

    for comp in comps:
        box = tuple(map(int, comp["bbox"]))
        assert bbox_h(box) == _OBJ_SIZE and bbox_w(box) == _OBJ_SIZE, "every object must span 4x4"
        r0, c0, r1, c1 = box
        sub = crop_bbox(g, box)
        color = int(comp["color"])

        if _is_solid(sub, color):
            kind = "solid"
        else:
            assert _is_frame(sub, color, bg), "every 4x4 object must be a solid or a hollow frame"
            kind = "frame"

        sequences.setdefault((c0, c1), []).append(
            {"kind": kind, "color": color, "bbox": (r0, c0, r1, c1)}
        )

    ordered_sequences: list[list[dict[str, object]]] = []
    for key in sorted(sequences):
        seq = sorted(sequences[key], key=lambda obj: int(obj["bbox"][0]))
        solid_count = sum(1 for obj in seq if obj["kind"] == "solid")
        assert solid_count == 1, "each vertical sequence must contain exactly one solid square"
        ordered_sequences.append(seq)

    assert ordered_sequences, "expected at least one vertical sequence"
    return ordered_sequences


def _simulate_sequence(
    sequence: list[dict[str, object]], height: int
) -> tuple[int, int, list[tuple[int, int]], int | None]:
    solid_idx = next(i for i, obj in enumerate(sequence) if obj["kind"] == "solid")
    current_color = int(sequence[solid_idx]["color"])
    passed: list[tuple[int, int]] = []
    stop_idx: int | None = None

    for idx in range(solid_idx + 1, len(sequence)):
        frame_color = int(sequence[idx]["color"])
        if frame_color == current_color:
            stop_idx = idx
            final_row = int(sequence[idx]["bbox"][0]) - _OBJ_SIZE
            break
        passed.append((idx, current_color))
        current_color = frame_color
    else:
        final_row = int(height) - _OBJ_SIZE

    return final_row, current_color, passed, stop_idx


def _validate_landing(
    sequence: list[dict[str, object]], final_row: int, stop_idx: int | None
) -> None:
    c0 = int(sequence[0]["bbox"][1])
    landing = (int(final_row), c0, int(final_row) + _OBJ_SIZE - 1, c0 + _OBJ_SIZE - 1)
    assert int(final_row) >= 0, "final landing row must stay inside the grid"

    solid_idx = next(i for i, obj in enumerate(sequence) if obj["kind"] == "solid")
    for idx, obj in enumerate(sequence):
        if idx == solid_idx:
            continue
        if stop_idx is not None and idx >= stop_idx:
            if idx == stop_idx:
                assert int(final_row) + _OBJ_SIZE == int(obj["bbox"][0]), "stopped solid must sit flush above the stop frame"
            continue
        assert not _boxes_overlap(landing, tuple(map(int, obj["bbox"]))), "final solid overlaps another object"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each 4x4 solid square downward through hollow frames while leaving relay-colored traces inside passed frames.

    Input structure:
    - The background is black.
    - Non-background pixels form several independent vertical 4x4-object sequences, each confined to one 4-column lane.
    - Every object is either a monochrome hollow frame with a black 2x2 interior or a monochrome solid 4x4 square, and each lane contains exactly one solid square.

    Rule:
    - Process each lane independently: remove the solid square from its original position, then let it fall straight down.
    - Whenever the falling solid reaches a lower hollow frame of a different border color, fill that frame's 2x2 interior with the solid's current color, then recolor the solid to the frame's border color and continue falling.
    - If the next lower frame has the same color as the solid's current color, the solid stops flush above that frame; if no such frame is encountered, it stops flush with the bottom edge.

    Key insight:
    - Passage depends on the solid's evolving current color, not its initial color, and each passed frame preserves the pre-change color as a trace fossil inside its hollow center.
    """
    g = np.asarray(grid, dtype=int)
    sequences = _extract_sequences(g)
    out = g.copy()

    for seq in sequences:
        solid_idx = next(i for i, obj in enumerate(seq) if obj["kind"] == "solid")
        solid_box = tuple(map(int, seq[solid_idx]["bbox"]))
        sr0, sc0, _sr1, _sc1 = solid_box
        out[sr0 : sr0 + _OBJ_SIZE, sc0 : sc0 + _OBJ_SIZE] = Color.BLACK

        final_row, final_color, passed, stop_idx = _simulate_sequence(seq, g.shape[0])
        _validate_landing(seq, final_row, stop_idx)

        for frame_idx, trace_color in passed:
            fr0, fc0, _fr1, _fc1 = map(int, seq[frame_idx]["bbox"])
            out[fr0 + 1 : fr0 + 3, fc0 + 1 : fc0 + 3] = int(trace_color)

        out[final_row : final_row + _OBJ_SIZE, sc0 : sc0 + _OBJ_SIZE] = int(final_color)

    return out


def _sample_sequence_layout(
    rng: np.random.Generator, height: int, count: int, min_gaps: list[int]
) -> list[int] | None:
    min_total = (_OBJ_SIZE * int(count)) + int(sum(min_gaps)) + _OBJ_SIZE
    slack = int(height) - min_total
    if slack < 0:
        return None

    extras = rng.multinomial(slack, np.full(count + 1, 1.0 / (count + 1)))
    top_margin = int(extras[0])
    gaps = [int(min_gaps[i] + extras[i + 1]) for i in range(count - 1)]

    rows = [top_margin]
    for gap in gaps:
        rows.append(int(rows[-1] + _OBJ_SIZE + gap))
    return rows


def _pick_color(rng: np.random.Generator, exclude: set[int]) -> int:
    choices = [color for color in _COLOR_POOL if int(color) not in exclude]
    return int(rng.choice(choices))


def _build_sequence_spec(
    rng: np.random.Generator, height: int, c0: int, branch: str
) -> dict[str, object] | None:
    if branch == "pass_then_stop":
        candidates = [(count, solid_idx) for count in _OBJECT_COUNT_OPTIONS for solid_idx in range(count - 2)]
    else:
        candidates = [(count, solid_idx) for count in _OBJECT_COUNT_OPTIONS for solid_idx in range(count - 1)]
    if not candidates:
        return None

    count, solid_idx = candidates[int(rng.integers(len(candidates)))]
    below = count - solid_idx - 1
    assert below >= 1

    pass_count = 0
    stop_idx: int | None = None
    min_gaps = [1] * (count - 1)

    if branch == "stop_first":
        stop_idx = solid_idx + 1
    elif branch == "pass_then_stop":
        pass_count = int(rng.integers(1, below))
        stop_idx = solid_idx + pass_count + 1
        min_gaps[stop_idx - 1] = 4
    else:
        assert branch == "pass_all"

    rows = _sample_sequence_layout(rng, height, count, min_gaps)
    if rows is None:
        return None

    solid_color = _pick_color(rng, set())
    object_colors = [None] * count
    object_colors[solid_idx] = solid_color

    for idx in range(solid_idx):
        object_colors[idx] = _pick_color(rng, set())

    current_color = solid_color
    for idx in range(solid_idx + 1, count):
        if stop_idx is not None and idx == stop_idx:
            object_colors[idx] = current_color
            break
        next_color = _pick_color(rng, {current_color})
        object_colors[idx] = next_color
        current_color = next_color

    for idx in range(solid_idx + 1, count):
        if object_colors[idx] is None:
            object_colors[idx] = _pick_color(rng, set())

    objects = []
    for idx, row in enumerate(rows):
        objects.append(
            {
                "row": int(row),
                "col": int(c0),
                "kind": "solid" if idx == solid_idx else "frame",
                "color": int(object_colors[idx]),
            }
        )

    final_row, _final_color, _passed, computed_stop = _simulate_spec_objects(objects, height)
    if branch == "stop_first":
        assert computed_stop == solid_idx + 1
    elif branch == "pass_then_stop":
        assert computed_stop == stop_idx and pass_count >= 1
    else:
        assert computed_stop is None

    if not _spec_landing_is_clear(objects, final_row, computed_stop):
        return None

    return {"branch": branch, "objects": objects}


def _simulate_spec_objects(
    objects: list[dict[str, int | str]], height: int
) -> tuple[int, int, list[tuple[int, int]], int | None]:
    solid_idx = next(i for i, obj in enumerate(objects) if obj["kind"] == "solid")
    current_color = int(objects[solid_idx]["color"])
    passed: list[tuple[int, int]] = []
    stop_idx: int | None = None

    for idx in range(solid_idx + 1, len(objects)):
        frame_color = int(objects[idx]["color"])
        if frame_color == current_color:
            stop_idx = idx
            final_row = int(objects[idx]["row"]) - _OBJ_SIZE
            break
        passed.append((idx, current_color))
        current_color = frame_color
    else:
        final_row = int(height) - _OBJ_SIZE

    return final_row, current_color, passed, stop_idx


def _spec_landing_is_clear(
    objects: list[dict[str, int | str]], final_row: int, stop_idx: int | None
) -> bool:
    c0 = int(objects[0]["col"])
    landing = (int(final_row), c0, int(final_row) + _OBJ_SIZE - 1, c0 + _OBJ_SIZE - 1)
    solid_idx = next(i for i, obj in enumerate(objects) if obj["kind"] == "solid")

    for idx, obj in enumerate(objects):
        if idx == solid_idx:
            continue
        box = (
            int(obj["row"]),
            int(obj["col"]),
            int(obj["row"]) + _OBJ_SIZE - 1,
            int(obj["col"]) + _OBJ_SIZE - 1,
        )
        if stop_idx is not None and idx >= stop_idx:
            if idx == stop_idx and int(final_row) + _OBJ_SIZE != int(obj["row"]):
                return False
            continue
        if _boxes_overlap(landing, box):
            return False
    return int(final_row) >= 0


def _render_input(specs: list[dict[str, object]], height: int, width: int) -> np.ndarray:
    grid = np.zeros((int(height), int(width)), dtype=int)
    for spec in specs:
        for obj in spec["objects"]:
            row = int(obj["row"])
            col = int(obj["col"])
            color = int(obj["color"])
            sprite = _solid_sprite(color) if obj["kind"] == "solid" else _frame_sprite(color)
            grid[row : row + _OBJ_SIZE, col : col + _OBJ_SIZE] = sprite
    return grid


def _render_output_from_specs(specs: list[dict[str, object]], height: int, width: int) -> np.ndarray:
    out = _render_input(specs, height, width)
    for spec in specs:
        objects = spec["objects"]
        solid_idx = next(i for i, obj in enumerate(objects) if obj["kind"] == "solid")
        solid = objects[solid_idx]
        sr0 = int(solid["row"])
        sc0 = int(solid["col"])
        out[sr0 : sr0 + _OBJ_SIZE, sc0 : sc0 + _OBJ_SIZE] = Color.BLACK

        final_row, final_color, passed, _stop_idx = _simulate_spec_objects(objects, height)
        for frame_idx, trace_color in passed:
            frame = objects[frame_idx]
            fr0 = int(frame["row"])
            fc0 = int(frame["col"])
            out[fr0 + 1 : fr0 + 3, fc0 + 1 : fc0 + 3] = int(trace_color)

        out[final_row : final_row + _OBJ_SIZE, sc0 : sc0 + _OBJ_SIZE] = int(final_color)
    return out


def _sample_columns(rng: np.random.Generator, seq_count: int) -> tuple[int, list[int]] | None:
    left_margin = int(rng.integers(*_MARGIN_RANGE))
    right_margin = int(rng.integers(*_MARGIN_RANGE))
    gaps = [int(rng.integers(*_SEQ_GAP_RANGE)) for _ in range(seq_count - 1)]

    cols = [left_margin]
    for gap in gaps:
        cols.append(int(cols[-1] + _OBJ_SIZE + gap))

    width = int(cols[-1] + _OBJ_SIZE + right_margin)
    if width > 30:
        return None
    return width, cols


def _branch_plan(rng: np.random.Generator, seq_count: int) -> list[str]:
    if seq_count >= 3:
        plan = ["pass_all", "pass_then_stop", "stop_first"]
        while len(plan) < seq_count:
            plan.append(str(rng.choice(["stop_first", "pass_then_stop", "pass_all"])))
    else:
        stop_branch = str(rng.choice(["stop_first", "pass_then_stop"]))
        plan = ["pass_all", stop_branch]
    rng.shuffle(plan)
    return plan


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(*_HEIGHT_RANGE))
    seq_count = int(rng.integers(*_SEQ_COUNT_RANGE))
    sampled = _sample_columns(rng, seq_count)
    if sampled is None:
        return None
    width, cols = sampled

    specs: list[dict[str, object]] = []
    for branch, c0 in zip(_branch_plan(rng, seq_count), cols):
        spec = _build_sequence_spec(rng, height, c0, branch)
        if spec is None:
            return None
        specs.append(spec)

    inp = _render_input(specs, height, width)
    out_from_spec = _render_output_from_specs(specs, height, width)

    parsed_sequences = _extract_sequences(inp)
    assert len(parsed_sequences) == seq_count, "parser must recover every generated lane"
    assert not np.array_equal(inp, out_from_spec), "generated sample must show a visible transformation"
    assert np.array_equal(main(inp), out_from_spec), "main() must match the independently rendered specification"
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid relay-gravity puzzle input")
