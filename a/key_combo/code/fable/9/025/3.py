from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import random
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _UTILS_DIR.exists() and str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import get_bg, in_bounds


RING_OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

SHAPES = {
    "h_bar": ((0, -1), (0, 1)),
    "v_bar": ((-1, 0), (1, 0)),
    "plus": ((-1, 0), (0, -1), (0, 1), (1, 0)),
    "x_corners": ((-1, -1), (-1, 1), (1, -1), (1, 1)),
}

GEN_HEIGHT_RANGE = (15, 23)
GEN_WIDTH_RANGE = (15, 25)
GEN_TRUE_TEMPLATE_RANGE = (1, 2)
GEN_FALSE_TEMPLATE_RANGE = (1, 2)
GEN_TRUE_COMPLETE_RANGE = (2, 3)
GEN_EXTRA_PARTIAL_RANGE = (0, 1)
GEN_CENTER_GAP = 4
GEN_MIN_TEMPLATE_DISTANCE = 6
GEN_PLACEMENT_TRIES = 120
GEN_ATTEMPTS = 400
BACKGROUND = 0


def _sorted_offsets(offsets) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(dr), int(dc)) for dr, dc in offsets))


VALID_OFFSET_SETS = {_sorted_offsets(offsets) for offsets in SHAPES.values()}


def _complete_signature(grid: np.ndarray, r: int, c: int, bg: int) -> tuple[int, int, tuple[tuple[int, int], ...]] | None:
    if not in_bounds(grid.shape, r, c):
        return None
    center = int(grid[r, c])
    if center == bg:
        return None

    shell_offsets: list[tuple[int, int]] = []
    shell_colors: list[int] = []
    for dr, dc in RING_OFFSETS:
        rr, cc = r + dr, c + dc
        if not in_bounds(grid.shape, rr, cc):
            return None
        value = int(grid[rr, cc])
        if value != bg:
            shell_offsets.append((int(dr), int(dc)))
            shell_colors.append(value)

    if not shell_offsets:
        return None
    if len(set(shell_colors)) != 1:
        return None
    shell = shell_colors[0]
    if shell == center:
        return None
    offsets = _sorted_offsets(shell_offsets)
    if offsets not in VALID_OFFSET_SETS:
        return None
    return center, shell, offsets


def _match_complete(
    grid: np.ndarray,
    r: int,
    c: int,
    center: int,
    shell: int,
    offsets: tuple[tuple[int, int], ...],
    bg: int,
) -> bool:
    if int(grid[r, c]) != int(center):
        return False
    wanted = set(offsets)
    for dr, dc in RING_OFFSETS:
        value = int(grid[r + dr, c + dc])
        if (dr, dc) in wanted:
            if value != int(shell):
                return False
        elif value != int(bg):
            return False
    return True


def _match_center_only(grid: np.ndarray, r: int, c: int, center: int, bg: int) -> bool:
    if int(grid[r, c]) != int(center):
        return False
    return all(int(grid[r + dr, c + dc]) == int(bg) for dr, dc in RING_OFFSETS)


def _match_shell_only(
    grid: np.ndarray,
    r: int,
    c: int,
    shell: int,
    offsets: tuple[tuple[int, int], ...],
    bg: int,
) -> bool:
    if int(grid[r, c]) != int(bg):
        return False
    wanted = set(offsets)
    for dr, dc in RING_OFFSETS:
        value = int(grid[r + dr, c + dc])
        if (dr, dc) in wanted:
            if value != int(shell):
                return False
        elif value != int(bg):
            return False
    return True


def _infer_templates(grid: np.ndarray, bg: int) -> tuple[dict[int, tuple[int, tuple[tuple[int, int], ...]]], Counter[int]]:
    templates: dict[int, tuple[int, tuple[tuple[int, int], ...]]] = {}
    counts: Counter[int] = Counter()

    for r in range(1, grid.shape[0] - 1):
        for c in range(1, grid.shape[1] - 1):
            sig = _complete_signature(grid, r, c, bg)
            if sig is None:
                continue
            center, shell, offsets = sig
            prev = templates.get(center)
            if prev is None:
                templates[center] = (int(shell), offsets)
            else:
                assert prev == (int(shell), offsets), "same center color must always encode the same shell"
            counts[center] += 1

    assert templates, "expected at least one complete motif"

    shell_to_center: dict[tuple[int, tuple[tuple[int, int], ...]], int] = {}
    for center, spec in templates.items():
        if spec in shell_to_center:
            assert shell_to_center[spec] == int(center), "shell-only copies must identify one center color"
        shell_to_center[spec] = int(center)
    return templates, counts


def _classify_copy(
    grid: np.ndarray,
    r: int,
    c: int,
    templates: dict[int, tuple[int, tuple[tuple[int, int], ...]]],
    bg: int,
) -> tuple[int, str] | None:
    matches: list[tuple[int, str]] = []
    value = int(grid[r, c])
    for center, (shell, offsets) in templates.items():
        if value == int(center):
            if _match_complete(grid, r, c, center, shell, offsets, bg):
                matches.append((int(center), "complete"))
            elif _match_center_only(grid, r, c, center, bg):
                matches.append((int(center), "center_only"))
        elif value == int(bg) and _match_shell_only(grid, r, c, shell, offsets, bg):
            matches.append((int(center), "shell_only"))
    assert len(matches) <= 1, "every visible copy must be uniquely classifiable"
    return matches[0] if matches else None


def _analyze_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    templates, complete_counts = _infer_templates(g, bg)

    copies: list[dict[str, object]] = []
    kind_counts: dict[int, Counter[str]] = defaultdict(Counter)
    covered = np.zeros(g.shape, dtype=int)

    for r in range(1, g.shape[0] - 1):
        for c in range(1, g.shape[1] - 1):
            match = _classify_copy(g, r, c, templates, bg)
            if match is None:
                continue
            center, kind = match
            shell, offsets = templates[center]
            footprint = [(int(r), int(c)), *[(int(r + dr), int(c + dc)) for dr, dc in offsets]]
            for rr, cc in footprint:
                if int(g[rr, cc]) != bg:
                    covered[rr, cc] += 1
            copies.append({"center": int(center), "kind": kind, "at": (int(r), int(c))})
            kind_counts[center][kind] += 1
            if kind == "complete":
                assert _complete_signature(g, r, c, bg) == (int(center), int(shell), offsets)

    fg = g != bg
    assert bool(fg.any()), "scene must contain foreground motifs"
    assert bool(np.all(covered[fg] == 1)), "every foreground pixel must belong to exactly one motif copy"

    return {
        "bg": bg,
        "templates": templates,
        "complete_counts": complete_counts,
        "kind_counts": kind_counts,
        "copies": copies,
    }


def _paint_full(out: np.ndarray, r: int, c: int, center: int, shell: int, offsets: tuple[tuple[int, int], ...]) -> None:
    out[r, c] = int(center)
    for dr, dc in offsets:
        out[r + dr, c + dc] = int(shell)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only motifs whose complete template repeats, completing their partial copies and erasing singleton-template families.

    Input structure:
    - A black background contains isolated 3x3 centered motifs: one center-color pixel and a same-color shell on selected ring cells.
    - Some copies are complete, while other copies are partial: either center-only or shell-only.
    - The same center color always implies the same shell color and shell offsets.

    Rule:
    - Count how many complete copies appear for each center color.
    - If a center color has at least two complete copies, it is a true template: keep its complete copies and complete all of its partial copies.
    - If a center color has exactly one complete copy, it is a fake template: erase that complete copy and every partial copy of the same template.

    Key insight:
    - A visible full motif is not automatically trustworthy; only repeated full appearances certify a template as valid.
    """
    scene = _analyze_scene(np.asarray(grid, dtype=int))
    g = np.asarray(grid, dtype=int)
    bg = int(scene["bg"])
    templates = scene["templates"]
    complete_counts = scene["complete_counts"]

    out = np.full_like(g, bg)
    for copy in scene["copies"]:
        center = int(copy["center"])
        if int(complete_counts[center]) < 2:
            continue
        r, c = copy["at"]
        shell, offsets = templates[center]
        _paint_full(out, r, c, center, shell, offsets)
    return out


def _lattice_capacity(height: int, width: int) -> int:
    usable_h = height - 2
    usable_w = width - 2
    return ((usable_h + GEN_CENTER_GAP - 1) // GEN_CENTER_GAP) * ((usable_w + GEN_CENTER_GAP - 1) // GEN_CENTER_GAP)


def _sample_templates() -> list[dict[str, object]]:
    n_true = random.randint(*GEN_TRUE_TEMPLATE_RANGE)
    n_false = random.randint(*GEN_FALSE_TEMPLATE_RANGE)
    if n_true + n_false > len(SHAPES):
        n_false = len(SHAPES) - n_true
    n_templates = n_true + n_false

    shape_names = random.sample(list(SHAPES), n_templates)
    colors = random.sample(list(range(1, 10)), 2 * n_templates)
    center_colors = colors[:n_templates]
    shell_colors = colors[n_templates:]

    templates: list[dict[str, object]] = []
    for i in range(n_templates):
        truth = i < n_true
        templates.append(
            {
                "id": i,
                "truth": truth,
                "center": int(center_colors[i]),
                "shell": int(shell_colors[i]),
                "offsets": _sorted_offsets(SHAPES[shape_names[i]]),
            }
        )
    return templates


def _build_copy_plan(templates: list[dict[str, object]]) -> list[dict[str, object]]:
    copies: list[dict[str, object]] = []

    true_ids = [int(t["id"]) for t in templates if bool(t["truth"])]
    false_ids = [int(t["id"]) for t in templates if not bool(t["truth"])]
    assert true_ids and false_ids

    must_have = {
        true_ids[0]: ["complete", "complete", "center_only", "shell_only"],
        false_ids[0]: ["complete", "center_only", "shell_only"],
    }

    for template in templates:
        tid = int(template["id"])
        base_roles = must_have.get(tid, [])
        if base_roles:
            roles = list(base_roles)
        elif bool(template["truth"]):
            roles = ["complete"] * random.randint(*GEN_TRUE_COMPLETE_RANGE)
        else:
            roles = ["complete"]

        if bool(template["truth"]) and tid not in must_have:
            roles += ["center_only"] * random.randint(*GEN_EXTRA_PARTIAL_RANGE)
            roles += ["shell_only"] * random.randint(*GEN_EXTRA_PARTIAL_RANGE)
        elif not bool(template["truth"]) and tid not in must_have:
            roles += ["center_only"] * random.randint(*GEN_EXTRA_PARTIAL_RANGE)
            roles += ["shell_only"] * random.randint(*GEN_EXTRA_PARTIAL_RANGE)
        else:
            roles += ["complete"] * random.randint(0, 1 if bool(template["truth"]) else 0)
            roles += ["center_only"] * random.randint(*GEN_EXTRA_PARTIAL_RANGE)
            roles += ["shell_only"] * random.randint(*GEN_EXTRA_PARTIAL_RANGE)

        random.shuffle(roles)
        for role in roles:
            copies.append({"template_id": tid, "kind": role})

    assert any(copy["kind"] == "center_only" for copy in copies if templates[int(copy["template_id"])]["truth"])
    assert any(copy["kind"] == "shell_only" for copy in copies if templates[int(copy["template_id"])]["truth"])
    assert any(copy["kind"] == "center_only" for copy in copies if not templates[int(copy["template_id"])]["truth"])
    assert any(copy["kind"] == "shell_only" for copy in copies if not templates[int(copy["template_id"])]["truth"])
    return copies


def _can_place_here(
    pos: tuple[int, int],
    placed: list[tuple[int, int]],
    copy: dict[str, object],
    complete_by_template: dict[int, list[tuple[int, int]]],
) -> bool:
    if any(max(abs(pos[0] - rr), abs(pos[1] - cc)) < GEN_CENTER_GAP for rr, cc in placed):
        return False
    if copy["kind"] != "complete":
        anchors = complete_by_template[int(copy["template_id"])]
        if not anchors:
            return False
        if min(abs(pos[0] - rr) + abs(pos[1] - cc) for rr, cc in anchors) < GEN_MIN_TEMPLATE_DISTANCE:
            return False
    return True


def _place_centers(height: int, width: int, copies: list[dict[str, object]]) -> list[tuple[int, int]] | None:
    candidates = [(int(r), int(c)) for r in range(1, height - 1) for c in range(1, width - 1)]
    order = sorted(range(len(copies)), key=lambda i: (copies[i]["kind"] != "complete", int(copies[i]["template_id"])))
    for _ in range(GEN_PLACEMENT_TRIES):
        positions: list[tuple[int, int] | None] = [None] * len(copies)
        placed: list[tuple[int, int]] = []
        complete_by_template: dict[int, list[tuple[int, int]]] = defaultdict(list)
        ok = True

        for idx in order:
            copy = copies[idx]
            feasible = [pos for pos in candidates if _can_place_here(pos, placed, copy, complete_by_template)]
            if not feasible:
                ok = False
                break
            pos = random.choice(feasible)
            positions[idx] = pos
            placed.append(pos)
            if copy["kind"] == "complete":
                complete_by_template[int(copy["template_id"])].append(pos)

        if ok:
            return [pos for pos in positions if pos is not None]
    return None


def _render_copy(grid: np.ndarray, template: dict[str, object], kind: str, pos: tuple[int, int]) -> None:
    r, c = pos
    center = int(template["center"])
    shell = int(template["shell"])
    offsets = template["offsets"]
    if kind in {"complete", "center_only"}:
        grid[r, c] = center
    if kind in {"complete", "shell_only"}:
        for dr, dc in offsets:
            grid[r + dr, c + dc] = shell


def _render_from_plan(
    shape: tuple[int, int],
    templates: list[dict[str, object]],
    copies: list[dict[str, object]],
    positions: list[tuple[int, int]],
    selector,
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    by_id = {int(t["id"]): t for t in templates}
    for copy, pos in zip(copies, positions):
        if selector(copy, by_id[int(copy["template_id"])]):
            template = by_id[int(copy["template_id"])]
            _paint_full(out, pos[0], pos[1], int(template["center"]), int(template["shell"]), template["offsets"])
    return out


def _validate_generated_scene(
    inp: np.ndarray,
    out: np.ndarray,
    templates: list[dict[str, object]],
    copies: list[dict[str, object]],
    positions: list[tuple[int, int]],
) -> bool:
    try:
        scene = _analyze_scene(inp)
    except Exception:
        return False

    by_id = {int(t["id"]): t for t in templates}
    planned_by_center = {int(t["center"]): t for t in templates}
    if set(scene["templates"]) != set(planned_by_center):
        return False

    planned_complete = Counter()
    planned_kind_counts: dict[int, Counter[str]] = defaultdict(Counter)
    for copy in copies:
        center = int(by_id[int(copy["template_id"])]["center"])
        planned_kind_counts[center][str(copy["kind"])] += 1
        if copy["kind"] == "complete":
            planned_complete[center] += 1

    for center, spec in scene["templates"].items():
        template = planned_by_center[int(center)]
        if spec != (int(template["shell"]), template["offsets"]):
            return False
        if int(scene["complete_counts"][center]) != int(planned_complete[center]):
            return False
        if scene["kind_counts"][center] != planned_kind_counts[center]:
            return False

    if not np.array_equal(main(inp), out):
        return False

    keep_everything = _render_from_plan(
        inp.shape,
        templates,
        copies,
        positions,
        lambda copy, _template: True,
    )
    keep_only_true_completes = _render_from_plan(
        inp.shape,
        templates,
        copies,
        positions,
        lambda copy, template: bool(template["truth"]) and str(copy["kind"]) == "complete",
    )
    if np.array_equal(keep_everything, out):
        return False
    if np.array_equal(keep_only_true_completes, out):
        return False
    return True


def _attempt() -> np.ndarray | None:
    height = random.randint(*GEN_HEIGHT_RANGE)
    width = random.randint(*GEN_WIDTH_RANGE)

    templates = _sample_templates()
    copies = _build_copy_plan(templates)
    if len(copies) > _lattice_capacity(height, width):
        return None

    positions = _place_centers(height, width, copies)
    if positions is None:
        return None

    inp = np.zeros((height, width), dtype=int)
    by_id = {int(t["id"]): t for t in templates}
    for copy, pos in zip(copies, positions):
        _render_copy(inp, by_id[int(copy["template_id"])], str(copy["kind"]), pos)

    out = _render_from_plan(
        inp.shape,
        templates,
        copies,
        positions,
        lambda copy, template: bool(template["truth"]),
    )
    if not _validate_generated_scene(inp, out, templates, copies, positions):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
