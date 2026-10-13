你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="eval" puzzle_id="581f7754" lines="8-22">
<description>
Slide multi-color sprites to align their marker pixels with matching isolated border anchors.

Input structure:
- bg = most common color.
- Multi-color sprites in the interior; each has a minority-color "marker" pixel.
- Isolated border pixels (non-bg with all 4-neighbors bg) act as anchors.

Rule:
- Match each sprite to the border anchor sharing its marker color.
- Top/bottom-edge anchor (non-corner): slide sprite horizontally so marker column = anchor column.
- Left/right-edge anchor (or corner): slide sprite vertically so marker row = anchor row.

Key insight: The anchor's edge position determines the slide axis — top/bottom means horizontal, left/right (or corner) means vertical.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="eval" puzzle_id="7b0280bc" lines="54-69">
<description>
Recolor the shortest path between two terminal squares in a square-and-track graph.

Input structure:
- bg = most common color.
- Two non-bg colors form solid NxN squares (N>=2): one color marks exactly 2 terminals, the other marks stations.
- A third non-bg color forms 1-cell-wide track segments connecting pairs of squares (graph edges).

Rule:
- Find the shortest path (fewest edges) between the two terminal squares.
- Recolor every station square on that path to green.
- Recolor every track segment along that path to grey.
- Everything else unchanged.

Key insight: The grid encodes a graph where squares are nodes and tracks are edges; the task is graph shortest-path, not spatial proximity.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="eval" puzzle_id="981571dc" lines="6-19">
<description>
Reconstruct a 4-way symmetric mosaic by filling missing (black) pixels.

Input structure:
- A colored grid exhibiting bilateral symmetry along both a vertical and a horizontal axis.
- Some pixels are erased to color 0 (black), creating gaps in the pattern.

Rule:
- Infer the vertical and horizontal reflection axes from the visible (non-zero) pixels.
- For each black pixel, recover its color from the mirrored position across either axis or both.
- When multiple mirrors yield candidates, use the majority color.

Key insight: The reflection axes are not given — they must be deduced from the surviving pixels by finding the axes that best preserve the existing symmetry.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="eval" puzzle_id="7b3084d4" lines="253-266">
<description>
Assemble colored rectangular sprites into a single tiled rectangle, anchored by a marker pixel.

Input structure:
- bg = most common color. Several solid-colored rectangular sprites scattered on the grid.
- One sprite has a unique single-pixel marker (different color) at one corner — this is the anchor.

Rule:
- Identify the anchor sprite (contains the marker pixel) and the remaining pieces.
- Rotate the anchor so the marker pixel lands at position (0,0) of the output.
- Pack all remaining pieces (allowing 90° rotations) to tile a complete rectangle with no gaps.

Key insight: The lone marker pixel picks the anchor and fixes the output orientation — it must sit at the top-left corner.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="eval" puzzle_id="67e490f4" lines="8-22">
<description>
Fill holes inside a large template frame using shape-matched outside sprites.

Input structure:
- bg = most common color. One large monochrome frame object (template).
- Inside the frame: holes of various shapes (non-frame-color regions).
- Outside the frame: small monochrome sprites scattered on bg.

Rule:
- Crop to the frame's bounding box; fill entirely with frame color.
- For each hole, find an outside sprite whose shape matches (rotation/flip allowed).
- Fill the hole with the matched sprite's color. If multiple sprites match, use the most frequent color.

Key insight: Sprites act as shape-coded color keys — their shape identifies which hole they fill, their color determines the fill color.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="eval" puzzle_id="e87109e9" lines="68-83">
<description>
Emit four teal beams from a square, turning at colored obstacles per a legend.

Input structure:
- Top band: grey (5) legend with colored vertical bars in cells; bar position (left or right) encodes turn direction.
- Below: main grid with bg, a solid teal (8) square, and colored obstacle blocks.

Rule:
- Remove the legend from the output.
- From the teal square, shoot a beam in each cardinal direction (as a sliding copy of the square).
- Each beam paints teal over bg cells as it moves.
- On hitting an obstacle of color C, consult the legend: bar on left → turn counterclockwise, bar on right → turn clockwise. Continue in the new direction.
- Beams stop at grid edges.

Key insight: The legend's bar position (left vs right within its cell) is the encoding for turn direction at each obstacle color.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="eval" puzzle_id="28a6681f" lines="8-23">
<description>
Simulate water pixels falling under gravity into basins formed by colored obstacles.

Input structure:
- bg = 0. Water pixels = color 1 (blue).
- Other colors form static solid structures (walls, steps, containers).
- Water starts in unstable positions above or beside basins.

Rule:
- Each water pixel falls straight down through bg cells.
- When blocked below, it slides diagonally through the unique open side cell.
- When resting on a surface, it slides horizontally toward the nearest gap where it can fall again.
- Process repeats until no water pixel moves.

Key insight: Water seeks the deepest reachable basin via diagonal and horizontal sliding, not just vertical fall.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="eval" puzzle_id="4a21e3da" lines="8-20">
<description>
Cut a sprite along border-control rays and slide pieces to grid corners.

Input structure:
- bg, fg (main object, ~centered), ctrl (1-2 pixels on grid edges).

Rule:
- Draw rays inward from each ctrl pixel along its row/col, painting only bg cells up to the farthest fg pixel (intermittent).
- Rays partition the grid; fg on the cut line stays in place.
- Cut pieces in regions touching a control edge slide to that corner; others vanish.

Key insight: Only regions sharing a border edge with a control pixel keep their piece; the ray skips fg cells.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="eval" puzzle_id="1ae2feb7" lines="30-45">
<description>
Extend row bars periodically across the opposite side of a separator column.

Input structure:
- bg = background. A near-solid vertical separator divides the grid into two sides.
- One side contains horizontal colored bars; the other side is blank target space.
- Bars may be stacked by row and may sit at different distances from the separator.

Rule:
- Identify the separator as the column whose non-bg cells are one color.
- Treat the denser side as the bar side and collect each contiguous row run.
- Process bars from nearest to farthest from the separator; for each bar, paint its
  color every bar-length cells into the blank side without overwriting earlier paint.

Key insight: Each bar's length is the period for its extension, and nearer bars claim target cells first.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="eval" puzzle_id="cbebaa4b" lines="51-65">
<description>
Assemble scattered jigsaw sprites around a central yellow anchor by pairing red connector pixels.

Input structure:
- bg = black (0). Several colored sprites scattered on the grid.
- One anchor sprite contains a 3×3 yellow (4) block.
- Each sprite has red (2) connector pixels along its edges.

Rule:
- The anchor stays in place; remaining sprites attach one at a time.
- Each sprite is translated so that one of its red pixels overlaps an unpaired red pixel on the growing assembly.
- Every red connector must pair with exactly one other, forming a complete joined structure.

Key insight: Red pixels act as jigsaw pegs — each must match exactly one counterpart, and the assembly order is determined by which unpaired connectors remain.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="eval" puzzle_id="d35bdbdc" lines="120-134">
<description>
Transfer center colors between bordered sprites matched by color, keeping only path-connected ones.

Input structure:
- bg, grey (5) path network, and several small "bordered sprites" each having a border color and a single center pixel of a different color.
- Grey path forms a one-stroke trail under 8-connectivity; its start/end cells touch exactly two destination sprites via ports.

Rule:
- Destination sprites are determined by the grey trail's two endpoints.
- For each destination, find the source sprite whose border color equals the destination's center color.
- Replace the destination's center pixel with the source's center color.
- Delete all non-destination sprites; grey paths and destinations remain.

Key insight: Center color of each destination acts as a lookup key into the border colors of other sprites, enabling indirect color transfer.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="eval" puzzle_id="a47bf94d" lines="178-189">
<description>
Transmit colored tokens along wire networks, converting each to its complementary 3×3 pattern.

Input structure:
- bg = background. Non-bg cells form wire networks and 3×3 tokens (solid blocks or X/diagonal patterns) near wire endpoints.

Rule:
- Every solid 3×3 token is replaced in-place by a diamond (center + 4 cardinal neighbors).
- Each token near a wire endpoint sends its complement to the opposite endpoint: solid emits X, X emits diamond.

Key insight: Token shape cycles solid → X → diamond; wires route each complement to the far end.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="eval" puzzle_id="58490d8a" lines="9-21">
<description>
Count colored objects and record tallies inside a legend box.

Input structure:
- A black rectangular legend box with key-color pixels in its leftmost non-black column.
- Outside the box, distinct objects of each key color are scattered across the grid.

Rule:
- For each key color, count separate same-colored objects outside the legend box.
- Output the legend box repainted: each key row gets one dot per object, spaced every other cell rightward from the key pixel.

Key insight: The legend is both the color index and the output canvas — object counts become dot sequences on the matching row.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="eval" puzzle_id="d59b0160" lines="39-54">
<description>
Delete connected components whose colors match a legend swatch.

Input structure:
- Background (bg) fills most of the grid.
- Top-left corner: a small square legend region bordered on its bottom and right edges by a single color; the interior holds a set of "target" colors.
- Elsewhere: multicolored objects (4-connected components) on the bg.

Rule:
- Read the target color set from inside the legend border.
- For each connected component, check whether it contains **all** target colors.
- Erase (fill with bg) every component that does; leave all others unchanged.
- Output grid is the same size.

Key insight: The legend acts as a filter specification — only components carrying the full color palette shown in the legend are removed.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="eval" puzzle_id="221dfab4" lines="6-19">
<description>
Propagate a repeating striped pattern from a yellow border segment toward the opposite edge, recoloring foreground objects on certain stripes.

Input structure:
- bg = most common color. Foreground blobs of another color scattered in the interior.
- A short yellow (4) segment sits on one border edge.

Rule:
- From the yellow segment, march toward the opposite edge in steps of 2 cells, cycling colors Y, Y, G (4, 4, 3). The 1-cell gap between stripes is forced to bg.
- Yellow stripes paint only within the segment's span.
- Green stripes additionally recolor all non-bg pixels across the entire row/column to green.

Key insight: Green stripes act as a "flood" that recolors existing objects on the full row/column, while yellow stripes are confined to the segment span.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="eval" puzzle_id="446ef5d2" lines="8-20">
<description>
Assemble scattered rectangular pieces into a rectangle at an L-marker corner.

Input structure:
- bg = most common color. Non-bg rectangular pieces scattered on the grid.
- 3-pixel L-marker (2x2 minus one cell) tags the corner; the gap is the corner cell.

Rule:
- rect_col = most common color across all piece pixels; all outer rectangle edges must be rect_col.
- Translate pieces (no rotation) to tile the rectangle; touching edges must match, with rect_col as wildcard.

Key insight: rect_col acts as wildcard during edge matching — only non-rect_col pixels at shared boundaries must agree.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="eval" puzzle_id="9aaea919" lines="38-51">
<description>
Transfer sprites from red-marked lanes to green-marked lanes, recoloring and stacking.

Input structure:
- bg = most common color. Red (2) and green (3) thin bars define vertical lanes.
- Each lane contains repeated identical monochrome sprites in an evenly spaced vertical grid.

Rule:
- Each red lane's sprites turn grey (5); their count transfers to the nearest green lane.
- New sprites fill empty lattice slots bottom-up, using the source shape recolored to the destination lane's sprite color.
- All marker bars are removed.

Key insight: Sprite count is conserved across lanes — red sprites become grey, and equally many new sprites stack into the green lane's regular grid.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="eval" puzzle_id="2d0172a1" lines="209-224">
<description>
Convert a rough nested line drawing into a clean canonical frame-and-dot diagram.

Input structure:
- The grid has one background color and one foreground color.
- Foreground 4-connected objects are rough closed frames or small filled blobs.
- Frames may contain other frames/blobs; blobs outside a frame remain outside it.

Rule:
- Detect the containment tree of foreground objects using holes of each frame object.
- Replace every blob with one foreground pixel.
- Replace every frame with a rectangular border padded two cells around its direct children.
- Place siblings by their original row/column overlap and order, with one empty cell between canonical objects.

Key insight: The exact wobbly geometry is irrelevant; only enclosure plus relative sibling positions define the output.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="eval" puzzle_id="dbff022c" lines="275-288">
<description>
Fill hollow sprites' interiors using a color-palette legend on the grid edge.

Input structure:
- bg = most common color. Several hollow sprites with distinct border colors.
- A small 2×N or N×2 palette block sits on one grid edge; one row/column lists border colors, the other lists corresponding fill colors.

Rule:
- Locate the palette legend on a grid edge and read it as a border→fill color mapping.
- Find enclosed bg regions (holes) inside sprites that don't touch the grid boundary.
- For each hole, identify the most common adjacent border color, look it up in the palette, and fill the hole with the mapped fill color.

Key insight: The palette's edge position and orientation determine reading direction; border colors in the palette uniquely key into fill colors.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="eval" puzzle_id="5dbc8537" lines="10-23">
<description>
Fill rectangular cavities in a template shape by placing source sprites into matching holes.

Input structure:
- bg = most common color. A dominant "base" color forms a large frame/template with rectangular gaps.
- Small colored sprites sit outside the template region.

Rule:
- The base color (largest connected component excluding bg) defines the template bbox.
- Each rectangular hole inside the template matches exactly one outside sprite by size.
- Place each sprite into its matching cavity; output is the filled template bbox.

Key insight: Sprite bounding boxes match cavity dimensions exactly — it's a jigsaw-fit puzzle where each piece has a unique slot.
</description>
</reference_puzzle>

你的任务：
以参考题为标杆，运用发散性思维进行创新扩展。请发挥创造力，构思全新且巧妙的推理谜题。用中文创造全新的 ARC 谜题描述。
创新方式包括但不限于：
**机制重组**：将多道参考题各自的核心机制提取出来，组合成一个新的多步管道，用到新的场景。
**机制变异**：取几道参考题的核心机制，以机制为基础进行创新并与其他多种机制进行创新重组，使解题路径发生质变而非量变。
**逆向构造**：取一道参考题的输出模式，思考"什么样的输入规则能产生类似的输出结构但需要完全不同的推理过程"。

严格使用下面的 XML 风格格式输出：

<new_puzzle>
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。
</new_puzzle>

请一次性生成 5 个 `<new_puzzle>`，若有较多合适的创新，也可以生成更多，不设上限，但不要输出额外解释。

注意事项：
1. 任务不应依赖大量先验知识或复杂计算，而应依赖对模式、结构与空间关系的直觉推理。
2. 关于"计算"的边界：
   - **允许使用**（这些都是人类能心算的视觉量）：数像素 / 孔洞 / 对象 / 邻居（≤10 量级）、比大小、按某个可见属性排序、奇偶判断、取最大 / 最小、用一个计数值作为另一个等量几何量（如 N 个像素 → 长度 N 的射线、面积 N → 边长 N）、判断接触 / 相邻 / 包含、粗略中心位置（簇的视觉中点）、"距离"作为"几步"而不是公式等人类能够直观感受计算的内容。
   - **禁止使用**（这些是算法量而非视觉量，会迫使解题者依赖公式而非视觉直觉）：曼哈顿 / 欧氏 / Chebyshev 距离作为精确查表键、凸包面积、二进制位编码 / 位运算、模运算（取模） / 余数、拓扑距离（BFS / 最短路 / 跳数）、精确几何质心坐标、凸角 / 凹角计数等复杂且不易被看出来的计算都是禁止使用的。
3. 问题必须能从少量示例中推断出来（通常为 3 对训练样例 + 1 个测试样例）。设计时从人类解题者的视觉角度思考：规则涉及的所有关键要素（对象、关系、属性）必须在网格上肉眼可见，不应依赖人眼无法直接观察到的隐含信息（例如颜色id和颜色亮度是禁止使用的）或需要推算的隐藏属性。规则背后的逻辑本身可以有一定挑战性，但解题所需的所有线索必须能从网格中直接获取。
4. 生成的问题描述旨在促进后续用于创建题目网格的代码生成。因此，对输入 / 输出网格的描述以及规则总结必须极其详细；语言风格应易于理解；对于特定对象（如箭头），必须在描述中明确其绘制方式。
5. 描述谜题时避免使用 0 或 1 这类颜色编号；应使用颜色名称，例如 "red"。可用颜色如下：Black、Blue、Red、Green、Yellow、Grey、Magenta、Orange、Azure、Maroon。
6. 5 个新题的核心推理机制应明显不同——即从观察样例到归纳规则的认知路径不同，而不仅仅是场景、对象或颜色不同。
7. 题目难度应来自机制的隐蔽性而非步骤数。用参考题作为难度标尺——如果你生成的题一个样例就能猜到规则，说明机制太透明，需要加深隐蔽性。
8. 优先从参考题的机制出发进行重组或推演，而非从物理隐喻出发。物理隐喻容易导致大量题目归结为一种模式，例如"沿方向移动到阻挡处停止"。
9. 题目的难度优先来自规则的**结构层数**，而不是单一步骤的复杂度。关于"结构层数"的判定标准：**只有解题者需要主动推理才能完成的步骤才算一层**。肉眼一扫即可获得的信息（如"找到缺口在哪一侧"、"数出像素个数"、"识别哪个对象最大"）不构成独立的推理层——它们只是变换的前置输入，不增加认知难度。真正的多层结构是指每一层都需要解题者发现一个非显然的规律。以两层结构为例：
   - ✗ 伪两层："找到缺口方向"→"把东西推到对面" — 第一步肉眼可见，实际只有一层推理。
   - ✓ 真两层："通过孔洞数量匹配颜色图例"→"用匹配到的颜色重绘对象" — 两步都需要非平凡的归纳。

   如果你发现自己写出的题只是"对每个对象，根据它的某个属性做某变换"的单步规则，请引入一个需要主动推理的前置步骤。**用堆叠 tie-break 规则或复杂数学公式来"加难度"是劣解；用多一层需要归纳的结构嵌套来"加难度"才是好解**。
10. 在符合ARC题目和提示词要求的基础上，鼓励尝试创新规则为多层结构的题目，不局限于两层，可以创新三层或四层结构的题目，但不可为了多结构层数违背要求。

生成后对生成的题执行以下四项审查（审查内容不需要保存到文件）：
1. 原创性
  新题不可与上述参考题相同或类似，不可只是替换颜色、方向、对象类型等表层要素。判断标准：新题的核心推理路径应与任何一道参考题不同。
2. 确定性
  规则完全确定，同一输入只能产生唯一输出，每一步操作都有明确定义。
3. 可实现性
  所有必要细节都已给出——对象如何生成、如何识别、如何变换、边界情况如何处理，读完描述后可以直接写代码。
4. ARC 合规性
  网格尺寸合理（≤30×30），使用颜色不超过 10 种，规则复杂度适中，人类能从 3-5 个样例中发现规律。没有使用上述注意事项第二条禁止使用的复杂计算。

