你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="eval" puzzle_id="b9e38dc0" lines="7-18">
<description>
Fill the region enclosed by a wall sprite and radiate color outward through its opening.

Input structure:
- bg = most common color. Wall = most frequent non-bg color, forming a partial enclosure (e.g. U-shape).
- A fill-color pixel sits inside the enclosed area. Other non-bg pixels may act as obstacles.
Rule:
- Flood-fill the enclosed area with the fill color, radiating outward through the wall's open end.
- Obstacles cast "shadows" along the radiate direction, blocking fill behind them.

Key insight: The wall endpoints define the opening direction; fill radiates outward through it, and obstacles block propagation along that same axis.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="train" puzzle_id="2e65ae53" lines="88-103">
<description>
Fill every lattice slot from the color assigned to that slot position by the other lattices.

Input structure:
- The background is the dominant color.
- The largest foreground color forms several one-pixel-thick rectangular lattice frames.
- The frame lines divide each lattice into the same row-by-column set of rectangular slots; some slots already contain one non-frame color.

Rule:
- Infer a color for each logical slot position from the colored slots across all lattice frames.
- Repaint every slot interior in every lattice with the inferred color for its row/column position.
- Keep the frame lines and all background outside frames unchanged.

Key insight:
- Slot sizes vary by frame, but the complete color pattern is indexed by lattice slot position rather than by absolute pixel size or location.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="train" puzzle_id="3ad05f52" lines="128-142">
<description>
Paint the maze corridor and framed rooms lying on shortest paths between the two colored endpoint components.

Input structure
- The grid has a dominant background, one wall color forming thin orthogonal frames, and one endpoint/fill color.
- The endpoint/fill color forms exactly two 4-connected solid rectangular components; their larger side lengths define the framed room sizes.

Rule
- Treat wall pixels as blocked and every other pixel as passable.
- Find every passable pixel that lies on a shortest 4-connected path between the two endpoint components.
- Also fill any framed square room of an endpoint-derived size that intersects that shortest-path mask.

Key insight
- Shortest paths identify the route through open connector corridors, while the wall-bordered square rooms need to be expanded from the route centerline to their full interior.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="train" puzzle_id="4e7e0eb9" lines="31-49">
<description>
Transform each 9x9 panel by mirroring across its center guide line or filling all four block slots from its lone exemplar.

Input structure
The grid is made of 9x9 panels separated by one-cell-wide grid lines. Each
panel has four monochrome 3x3 slots at the quadrant positions, with either a
full non-background center row, a full non-background center column, or no
guide line.

Rule
A panel with a full center column is reflected left-to-right, and a panel
with a full center row is reflected top-to-bottom. A panel with no guide line
has exactly one slot color that appears once and one slot color that appears
three times; recolor all four slots to the once-occurring slot color.

Key insight
The center guide line encodes the mirror axis, while panels without a guide
line are completed by copying the structurally unique slot color.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="eval" puzzle_id="9385bd28" lines="71-87">
<description>
Fill rectangles between diagonal marker pairs using a color legend.

Input structure:
- bg grid with a vertical legend at the bottom-left: each row is a (marker_color, fill_color) pair.
- Scattered L-shaped corner markers of each marker color define opposite corners of axis-aligned rectangles.
- Some markers are solid 2×2 blocks instead of corner pairs.

Rule:
- Read legend bottom-to-top; for each entry find all pixels of that marker color.
- Determine which two bbox corners carry L-shaped anchors — the *other* diagonal defines the fill region.
- Fill the diamond (all shortest-path cells between those two corners) with the legend's fill color.
- Solid 2×2 marker blocks are simply recolored to the fill color.
- Later legend entries are protected from earlier fills.

Key insight: The L-shaped anchors sit on the marker diagonal; the fill goes on the opposite diagonal of the bounding box.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="eval" puzzle_id="db0c5428" lines="32-51">
<description>
Expand a 3x3 block motif into a 5x5 block motif around its missing center.

Input structure:
- Teal background surrounds one 9x9 foreground motif.
- The motif is a 3x3 array of 3x3 blocks; the middle block is empty teal.
- Colored pixels immediately around the empty middle block encode the border
  of the missing center block.

Rule:
- Keep the original 9x9 motif in place.
- Copy each of the eight surrounding 3x3 blocks to the opposite outer slot
  of the implied 5x5 block grid, without rotating it.
- Fill the center block from the eight pixels around the hole; its center is
  the motif's majority foreground color.

Key insight: The visible 9x9 motif is the middle of a larger periodic
block-level pattern, and the empty center is reconstructed from its immediate
inner boundary.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="train" puzzle_id="5d588b4d" lines="10-28">
<description>
Expand the top-left horizontal bar into a wrapped sequence of increasing and decreasing foreground runs.

Input structure:
- The grid has a background color and exactly one foreground color.
- All foreground pixels form one contiguous horizontal bar in the top row.
- The bar starts at the left edge; its length is the control value n.

Rule:
- Build a one-dimensional sequence of foreground runs with lengths
  1, 2, ..., n, n-1, ..., 2, 1.
- Separate adjacent runs by one background pixel.
- Wrap the sequence row-major across the input width and pad the final row
  with background pixels.

Key insight:
The output preserves the input width; the bar length controls the run-length
diamond, and the apparent 2D pattern is just that 1D sequence wrapped into rows.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="train" puzzle_id="1190bc91" lines="36-49">
<description>
Complete a sparse grid by casting occluding diagonal rays from the longest run and filling the remaining diagonal-connected regions from the short bars.

Input structure
- Black-background grid with one unique longest 4-connected foreground object.
- The longest object is a horizontal or vertical multicolor run; all other foreground objects are shorter monochrome bars.

Rule
- On a blank canvas, process the longest run from left to right or top to bottom; each pixel emits four 45-degree rays that stop at the first already-painted pixel.
- Each remaining monochrome bar fills every still-blank pixel reachable from it by diagonal steps.

Key insight
- Earlier pixels in the long run occlude later diagonal rays, and those ray barriers partition the background into diagonal-connected regions for the short bars.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="eval" puzzle_id="4c3d4a41" lines="15-28">
<description>
Fit a grey bar into a grey frame by aligning projections with colored columns, then stack colors above.

Input structure:
- bg, a grey structural color forming two components: a large rectangular frame (slot) and a smaller bar with upward projections.
- Colored pixels inside the slot mark target columns.

Rule:
- Move the bar upward into the slot so its bottom edge sits one cell above the slot's interior bottom.
- Shift it horizontally so the bar's upward projections align column-wise with the colored pixels in the slot.
- Stack the colored pixels immediately above the bar projections in each column.

Key insight: The bar's projections act as keys that must align with the colored-column "locks" in the slot.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="train" puzzle_id="342ae2ed" lines="34-49">
<description>
Draw the missing diagonal connector segment between each pair of same-color rectangular objects.

Input structure:
- The grid has one dominant background color.
- Every foreground color appears as exactly two separated solid rectangular 4-connected objects.
- The two rectangles of each color are diagonally offset with facing corners on a 45-degree line.

Rule:
- Keep all existing rectangles unchanged.
- For each foreground color, connect the lower facing corner of the upper rectangle to the upper facing corner of the lower rectangle.
- Paint only the intervening background pixels on that diagonal, excluding the two endpoint pixels already in the rectangles.

Key insight:
- Same-color rectangles define the endpoints; the output completes the open diagonal between their nearest facing corners.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="train" puzzle_id="b74ca5d1" lines="157-172">
<description>
Swap each 5x5 sprite with its corner marker color and stamp same-marker composite masks into the matching corner.

Input structure:
- A 30x30 grid has one dominant background color.
- Non-background corner pixels are marker labels.
- Every other foreground color is the body of one 5x5 sprite containing exactly one pixel of a corner marker color.

Rule:
- For each sprite, recolor body pixels to the sprite marker color and recolor the marker pixel back to its body color.
- If a marker lies on the side facing its matching corner and another same-marker sprite has a body pixel at that same local cell, use that visible body color for the marker pixel.
- For each corner marker, overlay the masks of all sprites with that marker color into the corner 5x5 block and paint the visible composite with the marker color.

Key insight:
The corner pixels are labels that group 5x5 sprites by their embedded marker color; each labeled corner shows the grouped sprite silhouette.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="train" puzzle_id="4a1cacc2" lines="8-23">
<description>
Fill the rectangle from the lone foreground pixel to its nearest grid corner.

Input structure
- The grid is a uniform background field with exactly one foreground pixel.
- The foreground pixel's color is the color used for all added pixels.

Rule
- Find the grid corner with the smallest Manhattan distance to the foreground pixel.
- Recolor every pixel in the inclusive axis-aligned rectangle bounded by that
  corner and the foreground pixel to the foreground color.

Key insight
- The single pixel is one corner of the rectangle; the opposite corner is the
  nearest corner of the whole canvas.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="eval" puzzle_id="271d71e2" lines="9-22">
<description>
Move bordered containers toward same-colored destination markers, charging gray cells to orange.

Input structure:
- bg = 6 (magenta). Black-bordered (0) rectangles with gray (5) / orange (7) interiors.
- A head sprite (9) sits adjacent to one container edge; a matching destination sprite lies along the movement axis.

Rule:
- Each container translates toward its destination along the head-to-dest axis.
- Steps = min(head-to-dest distance, number of gray interior cells).
- Gray cells become orange from the head-facing side inward, one per step.

Key insight: Gray cell count caps movement — a container stops short if it runs out of gray cells to charge.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="eval" puzzle_id="e8686506" lines="172-185">
<description>
Assemble scattered colored fragments into a left-right symmetric rectangle, jigsaw-style.

Input structure:
- bg = most common color; remaining colors form small sprite fragments.
- Optional roles: 4 singleton pixels of one color mark corners; horizontal bars mark top/bottom edges; paired vertical lines mark left/right edges.

Rule:
- Collect all non-bg connected components as jigsaw pieces.
- Determine output rectangle dimensions from total piece area.
- Pack pieces into the rectangle so it is exactly mirror-symmetric (left-right) and same-color pieces never 4-touch.

Key insight: Left-right symmetry plus the no-same-color-touching constraint make the assembly unique; corner/edge markers anchor it when present.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="train" puzzle_id="2601afb7" lines="30-45">
<description>
Rotate each vertical bar's color one slot right and its height one slot left.

Input structure
- A dominant background surrounds several monochrome one-cell-wide vertical bars.
- Every bar is bottom-aligned and occupies a distinct column.
- The ordered bar slots are read from left to right.

Rule
- Keep the grid shape and the bar columns unchanged.
- At slot i, use the color from the previous slot cyclically.
- At slot i, use the height from the next slot cyclically.

Key insight
- Color and height are independent cyclic attributes of the ordered bar slots:
  colors rotate right while heights rotate left.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="eval" puzzle_id="16de56c4" lines="9-34">
<description>
Fill periodic progressions that shoot from a unique grid edge.

Input structure:
- bg = most common color.  Sparse non-bg dots of several colors.
- All repeating colors originate from the same grid edge (top, bottom,
  left, or right — the one with the most non-bg dots).
- Each repeating color has exactly 2 dots per shooting-direction line;
  their gap defines the step.
- Every non-bg dot lies on a shooting-direction line that contains a
  repeating color.

Rule:
- Determine the unique shooting edge; left/right → rows, top/bottom → columns.
- For each shooting-direction line, the repeating color has exactly 2 dots;
  step = gap between them, offset = first position mod step.
- A different-colored dot aligned to the same beat (pos % step == offset) is
  a companion: it absorbs the ray and becomes the fill color, limiting the
  fill range to the span between repeating and companion dots.
- An unaligned dot is a bystander and left untouched.
- Without a companion the repeating color fills the entire line.

Key insight: the shooting edge is identified by counting boundary dots;
the gap between the two marker dots is the period; a companion sharing the
modular class acts as a wall that stops the ray and dictates the paint color.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="eval" puzzle_id="35ab12c3" lines="73-88">
<description>
Complete outlined shapes and carry attached shadow pixels along their edges.

Input structure:
- bg = background. Each active color marks vertices of an 8-direction aligned outline.
- A start vertex can have adjacent non-bg, different-colored shadow pixels.
- Other colored pixels may be distractors unless they lie on valid drawable edges.

Rule:
- For each color, order its pixels by polar angle around their centroid to form a cycle.
- Draw only horizontal, vertical, or 45-degree edges whose interiors are clear.
- From the chosen start vertex, copy adjacent shadow offsets along each drawable edge
  when the parallel path is in bounds and clear.

Key insight: The main outline is reconstructed first, then the start vertex's neighbor offsets define parallel shadow strokes.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="train" puzzle_id="f18ec8cc" lines="71-86">
<description>
Reorder the vertical color-major blocks while preserving every block's internal pixels.

Input structure
The grid is divided into vertical blocks, each with a unique dominant color.
Small marker pixels from other block colors may appear inside the blocks.

Rule
If adjacent shared-color marker rows read exactly as the current block order
from right to left, reverse the blocks. Otherwise move the leftmost block to
the far right and shift the remaining blocks left.

Key insight
The transformation acts on whole vertical blocks, so marker pixels move with
their source block instead of being recolored or independently rearranged.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="train" puzzle_id="320afe60" lines="41-56">
<description>
Slide each foreground object horizontally to the edge indicated by whether its shape has an open notch.

Input structure:
- Yellow is the background and there is one blue foreground color.
- The foreground is split into separated 4-connected objects.
- Each object keeps its row span and shape in the output.

Rule:
- If an object's bbox border contains a background gap flanked by object pixels along that same border, slide it to the right edge and recolor it green.
- Otherwise slide the object to the left edge and recolor it red.
- Start from a blank background canvas; no moved objects overlap.

Key insight:
- The target side is encoded by the object's shape, not by its original position: open notches go right, while closed or solid shapes go left.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="train" puzzle_id="afe3afe9" lines="106-119">
<description>
Convert each hollow 3x3 symbol to one pixel and slide the pixels clockwise from the marked edge.

Input structure
- The grid has a background, one full-edge marker line, and monochrome hollow 3x3 foreground symbols.
- The symbols lie on a coarse row/column lattice; fully empty lattice rows or columns are absent from the output.

Rule
- Replace each hollow 3x3 symbol with one pixel of its color in the corresponding coarse-grid cell.
- The marker edge gives the compaction direction by clockwise border travel: top -> right, right -> down, bottom -> left, left -> up.
- Preserve the order of non-background pixels while sliding them as far as possible in that direction.

Key insight
- The large picture is a scaled symbolic grid, and the blue edge marker acts as a gravity direction rather than as output content.
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
   - **不建议使用**（这些是算法量而非视觉量，会迫使解题者依赖公式而非视觉直觉）：曼哈顿 / 欧氏 / Chebyshev 距离作为精确查表键、凸包面积、二进制位编码 / 位运算、模运算 / 余数、拓扑距离（BFS / 最短路 / 跳数）、精确几何质心坐标、凸角 / 凹角计数等复杂且不易被看出来的计算。
3. 问题必须能从少量示例中推断出来（通常为 3 对训练样例 + 1 个测试样例）。设计时从人类解题者的视觉角度思考：规则涉及的所有关键要素（对象、关系、属性）必须在网格上肉眼可见，不应依赖人眼无法直接观察到的隐含信息（例如颜色id是不可用的）或需要推算的隐藏属性。规则背后的逻辑本身可以有一定挑战性，但解题所需的所有线索必须能从网格中直接获取。
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
  网格尺寸合理（≤30×30），使用颜色不超过 10 种，规则复杂度适中，人类能从 3-5 个样例中发现规律。
