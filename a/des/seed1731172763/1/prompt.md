你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="eval" puzzle_id="58f5dbd5" lines="20-34">
<description>
Carve color-matched motifs into solid blocks and pack them into a compact grid.

Input structure:
- bg = most common color.
- Several solid 5×5 monochrome blocks, each a distinct non-bg color.
- For each block color, a small same-colored motif sprite elsewhere on the grid.

Rule:
- Match each motif to its same-colored 5×5 block.
- Center the motif within the block and recolor those cells to bg (punching holes).
- Pack all carved blocks into an output grid preserving their original row/column order, separated by 1-cell gaps and a 1-cell outer border.

Key insight: The motif acts as a stencil — it's not stamped onto the block but carved out of it as background holes.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="train" puzzle_id="30f42897" lines="51-66">
<description>
Repeat the single colored border segment around the rectangular border, leaving one segment-length gap between copies.

Input structure:
- The background is the dominant color.
- There is exactly one foreground color.
- All foreground pixels form one contiguous segment on the rectangular border, possibly crossing the cycle start.

Rule:
- Unwrap the grid border clockwise into a cyclic path.
- Let n be the length of the foreground segment.
- Copy the segment with the same color every 2*n border positions around the cycle.

Key insight:
- The foreground segment is not extended by rows or columns directly; it is a motif on the perimeter cycle, repeated with an equal-sized gap.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="train" puzzle_id="18447a8d" lines="33-48">
<description>
Move each right-side colored shape into the left slot whose template color is its exact complement.

Input structure:
- The grid has two equal-width slots separated by one background column.
- Non-background rows form horizontal bands separated by all-background rows.
- Each band has a shared template color in the left slot and one colored source shape in the right slot.

Rule:
- For each band, define its destination shape as the cells in the left slot not occupied by the template color.
- Normalize every right-side source shape to right-slot coordinates and match it to the unique equal destination shape.
- Clear the original right-side source pixels and paint the matched left-slot complement with the source color.

Key insight:
The right-side objects are not placed by row order; each one completes the unique left template whose missing cells have the same shape.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="train" puzzle_id="3cd86f4f" lines="6-18">
<description>
Shear the rectangular input down-left onto a black canvas.

Input structure
A rectangular grid of colored pixels with no internal empty region to infer.

Rule
Create a black output with the same height and width W + H - 1, then copy
each input row r unchanged starting at column H - 1 - r.

Key insight
The colors and row contents are preserved; only the row offsets change so
the left edge forms a descending diagonal.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="train" puzzle_id="79cce52d" lines="29-44">
<description>
Crop off the control row and column, then cyclically roll the payload by the two marker offsets.

Input structure
- The grid is square, with a control row on top and a control column on the left.
- The control row and column share one filler color and one single-pixel marker color.
- The lower-right square is the payload and contains none of the control colors.

Rule
- Remove the top control row and left control column.
- Roll the remaining payload down by the marker's zero-based position in the left control column.
- Roll the payload right by the marker's zero-based position in the top control row.

Key insight
- The two border markers are cyclic shift amounts for the payload, not colors to preserve.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="train" puzzle_id="5a719d11" lines="44-59">
<description>
Within each horizontal pair of two-color panels, swap the motif masks and recolor each pasted mask with the source panel's base color.

Input structure:
- A separator color forms full blank rows and at least one full blank column.
- The non-separator area is split into row bands, each containing exactly two same-shaped rectangular panels.
- Each panel has a dominant base color and one minority motif color.

Rule:
- For every row band, clear each panel to its own base color.
- Paste the right panel's motif mask into the left panel using the right base color.
- Paste the left panel's motif mask into the right panel using the left base color.

Key insight:
- The motif colors themselves are distractors; only each motif's shape mask and the two panel base colors are exchanged.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="train" puzzle_id="2601afb7" lines="30-45">
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

<reference_puzzle index="8" source="train" puzzle_id="d255d7a7" lines="99-114">
<description>
Move the patterned edge band to the opposite edge along every row or column whose band slice contains the trail color.

Input structure
- The dominant color is the background.
- The two foreground colors are a frequent trail color and a rarer marker color.
- All marker pixels lie nearest to one grid edge, and that edge contains trail-color runs that define the source-band thickness.

Rule
- Choose the source edge identified by the marker pixels.
- A row or column perpendicular to that edge moves if its source-band slice contains the trail color.
- Copy that slice unchanged to the opposite edge, erase the remaining trail cells to background, and keep marker-only lines fixed.

Key insight
- The marker color identifies the active edge, while the trail color identifies which parallel rows or columns carry an edge-band slice across the grid.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="train" puzzle_id="a834deea" lines="20-39">
<description>
Fill each black 5x5 region's missing interior positions with a fixed 3x3 positional color legend.

Input structure
- The grid uses teal as the dominant background and black for one or more
  5x5 regions.
- Each black region has a black one-pixel border and a 3x3 interior that
  may already contain teal pixels.

Rule
- Overlay the legend 1 7 6 / 4 8 5 / 2 9 3 on the 3x3 interior of every
  5x5 black region.
- Wherever an interior non-center legend position is still black, recolor it
  to the corresponding legend color; leave existing teal pixels, the center,
  and the black border unchanged.

Key insight
- The colored output pixels are not derived from the teal motif shape; they
  are the fixed position labels for the missing cells of each 3x3 interior.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="train" puzzle_id="41ace6b5" lines="7-26">
<description>
Normalize vertical two-color bars between paired marker columns around the marker rows.

Input structure
- The background is the dominant color.
- A row of red marker pixels sits directly above a row of gray marker pixels
  in the same alternating columns.
- The columns between adjacent markers contain vertical bars made from a
  contiguous teal run above a contiguous blue run.

Rule
- Keep the marker pixels and all non-bar background cells fixed.
- Let h be the maximum teal-run height among the bar columns; paint teal in
  every bar column for h rows ending at the red marker row.
- In each bar column, move its blue run to start at the gray marker row and
  fill all lower cells in that column with maroon.

Key insight
- The marker rows are a baseline: teal is normalized into a common upper
  band, while each blue run keeps its own height below the baseline.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="train" puzzle_id="2f767503" lines="66-81">
<description>
Erase each target-color object crossed by the ray pointed by the bar-and-control marker.

Input structure:
- The background is the dominant color.
- There is one three-pixel straight marker bar and one singleton control pixel adjacent to the bar center.
- The remaining non-background color forms multiple 4-connected target objects.

Rule:
- Cast a straight ray from the marker bar center in the direction away from the adjacent control pixel.
- Find every 4-connected target-color object that intersects that ray.
- Recolor all pixels of those hit target objects to the background; all other pixels stay unchanged.

Key insight:
- The whole object is erased when the center ray touches any of its pixels; the full marker bar width is not swept.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="eval" puzzle_id="db695cfb" lines="8-19">
<description>
Connect same-color diagonal pixel pairs, then emit perpendicular rays from differently-colored pixels on those diagonals.

Input structure:
- bg = most common color. Line color (e.g. blue) has pixel pairs aligned on shared diagonals. Trigger color (e.g. magenta) pixels sit along those diagonals.

Rule:
- Draw diagonal lines connecting each pair of same-color aligned pixels.
- Where the diagonal crosses a trigger pixel, emit perpendicular diagonal rays (in the trigger's color) extending to the grid boundary.

Key insight: Trigger pixels act as beam-splitters — the perpendicular rays rotate 90° from the connecting diagonal's orientation.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="train" puzzle_id="f0100645" lines="51-68">
<description>
Slide each inner colored object horizontally into the same-colored vertical wall until it touches an obstacle.

Input structure
- The background is the dominant color.
- The only foreground colors are two non-background colors.
- One foreground color fills the left column and the other fills the right column.
- Extra 8-connected objects of each wall color appear away from that color's wall.

Rule
- Clear the extra objects from their original positions.
- For each wall color, process its 8-connected extra objects from nearest to farthest from the matching wall.
- Slide each object horizontally toward its matching wall as far as possible without overlapping the wall or an already settled object of the same color.
- Keep the two vertical wall columns fixed.

Key insight
- The side columns act like color-specific gravity walls: same-colored objects compact sideways against their own wall while preserving each object's shape.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="eval" puzzle_id="a32d8b75" lines="262-275">
<description>
Assemble and place enlarged sprites on a canvas using legend-panel instructions.

Input structure:
- Color-6 separators divide the grid into a left legend column and a main canvas.
- Each legend panel has (top to bottom): a unit sprite, a frame template, a 5x5 bordered square with a position-marker pixel, and a 3x3 rotation clock-hand.

Rule:
- Tile the unit sprite into each active cell of the frame to build an enlarged composite.
- Rotate the composite per the clock-hand direction (up=0, right=90, down=180, left=270 degrees).
- Place it on the canvas anchored at the position encoded by the marker pixel's row/col in the bordered square's 3x3 inner grid.

Key insight: The marker pixel's position in a 3x3 grid jointly encodes vertical and horizontal anchor alignment (start/middle/end).
</description>
</reference_puzzle>

<reference_puzzle index="15" source="train" puzzle_id="996ec1f3" lines="16-31">
<description>
Compress the noisy four-region board split by a solid cross into a 3x3 color map.

Input structure:
- A single full row and a single full column have the same solid separator color.
- The separator cross divides the input into four nonempty rectangular regions.
- Each region has a unique dominant color, with other colors acting as noise.

Rule:
- Find the unique solid same-color row/column pair.
- Fill the middle row and middle column of the 3x3 output with the separator color.
- Put the unique dominant color of each quadrant into the corresponding corner.

Key insight:
- The input grid is a noisy enlarged version of the desired 3x3 layout; the cross marks the coarse cell boundaries.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="eval" puzzle_id="67e490f4" lines="8-22">
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

<reference_puzzle index="17" source="train" puzzle_id="878187ab" lines="10-24">
<description>
Replace the scattered foreground markers with a bottom-left count rectangle whose two corner diagonals encode the marker counts.

Input structure
- A background grid with exactly two foreground colors used as scattered marker pixels.
- The two foreground colors have distinct pixel counts.

Rule
- Ignore marker positions and use only the two foreground counts.
- Make a 16-by-16 background canvas, fill a bottom-left red rectangle whose height is the smaller count and width is the larger count.
- Paint yellow pixels along the two rising 45-degree diagonals from the rectangle's bottom-left and bottom-right corners.

Key insight
- The marker colors and positions vary between examples; their counts alone determine the output rectangle dimensions.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="eval" puzzle_id="8698868d" lines="9-23">
<description>
Match sprites to frames by connected-hole count, then nest each sprite inside its frame.

Input structure:
- bg = azure (8). Large hollow squares (frames) arranged in a grid; smaller filled squares (sprites) placed elsewhere.
- Each frame and sprite has a distinct color. Sprites may contain holes (non-sprite-color pixels).

Rule:
- Frames form a regular grid; some may be bg-colored (invisible).
- Count connected hole regions inside each frame's interior and each sprite.
- Pair each frame with the sprite sharing the same hole count.
- Place the matched sprite inside the frame border; recolor sprite holes to the frame's color.

Key insight: The matching key is the number of connected hole components, not hole area or color.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="train" puzzle_id="aa62e3f4" lines="18-35">
<description>
Replace the outlined object with the outside contour of that outline in the marker color.

Input structure:
- A dominant background surrounds one 8-connected foreground outline.
- The outline is the unique largest foreground color.
- One or more smaller interior colors are inside the outline; the marker color
  is the unique smallest non-outline color.

Rule:
- Erase all original foreground pixels to background.
- Paint every background pixel that is 4-adjacent to the outline with the
  marker color.
- Keep the grid shape unchanged.

Key insight: The colored interior marker supplies the paint color, while the
largest outline supplies only the contour whose outside edge is traced.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="train" puzzle_id="1efba499" lines="70-85">
<description>
Swap paired marker pixels across the largest foreground-color barrier, placing them just outside the barrier segment.

Input structure:
- The background is the dominant color.
- One foreground color is the unique most frequent non-background color; its pixels form the stationary barrier.
- Other foreground pixels are single markers, sometimes paired on opposite sides of one contiguous barrier segment in the same row or column.

Rule:
- Scan each row and column that crosses exactly one contiguous barrier segment.
- If the line has exactly one marker on each side of that segment, erase both source markers and place each marker in the cell immediately beyond the opposite side of the segment.
- Lines with markers on only one side are unchanged; the barrier remains unchanged.

Key insight:
- The marker pairing is local to a row or column, and the destination is defined by the nearest cells outside the intervening barrier segment, not by the original distance from the barrier.
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
