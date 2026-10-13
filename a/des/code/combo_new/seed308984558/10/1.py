import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
MARKER_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
STYLES = ["full", "top", "bottom", "left", "right", "tl", "tr", "bl", "br"]


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(cells)
    return out


def _bbox(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _decode_style(mask: np.ndarray):
    if np.all(mask[[0, -1], :] == 0) and np.all(mask[:, [0, -1]] == 0):
        return "full"
    if np.all(mask[0, 1:-1] == 0):
        return "top"
    if np.all(mask[-1, 1:-1] == 0):
        return "bottom"
    if np.all(mask[1:-1, 0] == 0):
        return "left"
    if np.all(mask[1:-1, -1] == 0):
        return "right"
    if mask[0, 0] == 0:
        return "tl"
    if mask[0, -1] == 0:
        return "tr"
    if mask[-1, 0] == 0:
        return "bl"
    if mask[-1, -1] == 0:
        return "br"
    return "full"


def _apply_style(out: np.ndarray, r0: int, c0: int, r1: int, c1: int, style: str):
    out[r0, c0 : c1 + 1] = GREY
    out[r1, c0 : c1 + 1] = GREY
    out[r0 : r1 + 1, c0] = GREY
    out[r0 : r1 + 1, c1] = GREY
    if style == "top":
        out[r0, c0 + 1 : c1] = BLACK
    elif style == "bottom":
        out[r1, c0 + 1 : c1] = BLACK
    elif style == "left":
        out[r0 + 1 : r1, c0] = BLACK
    elif style == "right":
        out[r0 + 1 : r1, c1] = BLACK
    elif style == "tl":
        out[r0, c0] = BLACK
    elif style == "tr":
        out[r0, c1] = BLACK
    elif style == "bl":
        out[r1, c0] = BLACK
    elif style == "br":
        out[r1, c1] = BLACK


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each grey frame border by the legend template selected by its marker color.

    Input structure:
    - Grey hollow frames on black each contain one colored marker pixel.
    - A bottom legend row contains one small border template per marker color; each
      template is drawn in its own color and encodes a missing side or corner.

    Rule:
    - Read each colored legend template as a border style instruction.
    - Remove the legend area from the output.
    - For every main grey frame, keep its bounding box and marker pixel, but redraw the
      grey border using the selected template's missing side/corner pattern.

    Key insight:
    - Marker color is only a lookup key: the useful information is the legend border
      geometry associated with that color, which must be transferred to larger frames.
    """
    legend_h = 5
    legend = grid[-legend_h:]
    styles = {}
    for color in [int(v) for v in np.unique(legend) if v != BLACK]:
        for comp in _components(legend == color):
            r0, c0, r1, c1 = _bbox(comp)
            patch = legend[r0 : r1 + 1, c0 : c1 + 1]
            mask = (patch == color).astype(int)
            styles[color] = _decode_style(mask)
    out = grid[:-legend_h].copy()
    for comp in _components(out == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        markers = [int(v) for v in np.unique(inner) if v not in (BLACK, GREY)]
        if len(markers) != 1:
            continue
        marker = markers[0]
        _apply_style(out, r0, c0, r1, c1, styles.get(marker, "full"))
    return out


def generate_input() -> np.ndarray:
    while True:
        colors = random.sample(MARKER_COLORS, random.randint(3, 4))
        chosen = random.sample(STYLES, len(colors))
        frames = []
        cur_c = 1
        max_h = 0
        for _ in colors:
            ih = random.randint(3, 6)
            iw = random.randint(3, 6)
            fh, fw = ih + 2, iw + 2
            frames.append((fh, fw))
            cur_c += fw + random.randint(1, 3)
            max_h = max(max_h, fh)
        h = max_h + 7
        w = max(cur_c + 1, len(colors) * 6 + 2)
        if h > 30 or w > 30:
            continue
        grid = np.zeros((h, w), dtype=int)
        c = 1
        for color, style in zip(colors, chosen):
            patch = np.zeros((4, 4), dtype=int)
            patch[0, :] = color
            patch[-1, :] = color
            patch[:, 0] = color
            patch[:, -1] = color
            if style == "top":
                patch[0, 1:-1] = BLACK
            elif style == "bottom":
                patch[-1, 1:-1] = BLACK
            elif style == "left":
                patch[1:-1, 0] = BLACK
            elif style == "right":
                patch[1:-1, -1] = BLACK
            elif style == "tl":
                patch[0, 0] = BLACK
            elif style == "tr":
                patch[0, -1] = BLACK
            elif style == "bl":
                patch[-1, 0] = BLACK
            elif style == "br":
                patch[-1, -1] = BLACK
            grid[-5:-1, c : c + 4] = patch
            c += 5
        c0 = 1
        ok = True
        for color, (fh, fw) in zip(colors, frames):
            if c0 + fw > w:
                ok = False
                break
            r0 = random.randint(0, max_h - fh)
            grid[r0, c0 : c0 + fw] = GREY
            grid[r0 + fh - 1, c0 : c0 + fw] = GREY
            grid[r0 : r0 + fh, c0] = GREY
            grid[r0 : r0 + fh, c0 + fw - 1] = GREY
            mr = random.randint(r0 + 1, r0 + fh - 2)
            mc = random.randint(c0 + 1, c0 + fw - 2)
            grid[mr, mc] = color
            c0 += fw + random.randint(1, 3)
        if ok:
            return grid
