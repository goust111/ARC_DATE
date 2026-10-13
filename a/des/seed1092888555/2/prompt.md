你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="eval" puzzle_id="fc7cae8d" lines="9-21">
<description>
Align a central rectangle to two colored frame-edge controls via rotation/reflection.

Input structure:
- bg grid with two solid-colored border edges (distinct non-bg colors).
- A central filled rectangle with 1px colored guide edges on two sides, colors matching the frame controls.

Rule:
- Rotate/mirror the object so its guide edges face the same sides as the matching frame controls.
- Crop off the 1px guide edges from aligned sides.

Key insight: Guide-edge colors encode target frame sides; the correct dihedral transform matches both simultaneously.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="train" puzzle_id="396d80d7" lines="18-30">
<description>
Paint the inner foreground color into every background pixel that touches the enclosing frame only diagonally.

Input structure:
- The grid has one dominant background color and exactly two foreground colors.
- One foreground color is an enclosing frame: its bbox strictly contains the bbox of the other, inner color.

Rule:
- Keep all existing pixels unchanged.
- For each background pixel, if it is diagonally adjacent to the frame color and has no orthogonal adjacency to the frame color, recolor it to the inner color.

Key insight: The inner color marks the color to use, while the frame's diagonal-only background neighbors define the cells to add.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="train" puzzle_id="252143c9" lines="14-33">
<description>
Keep only the center color and redraw it as a diagonal ray to its indicated corner.

Input structure:
- The grid is an odd square with a unique dominant background color.
- The center pixel is foreground.
- Every off-center pixel with the center color lies in the same diagonal
  quadrant relative to the center.

Rule:
- Fill the whole output with the background color.
- Use the quadrant of the center-colored pixels to choose one diagonal
  direction from the center.
- Draw a continuous line of the center color from the center to the grid
  edge in that direction.

Key insight:
The color at the exact center identifies the surviving color, while the
other pixels of that color identify which corner the center should connect to.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="train" puzzle_id="522fdd07" lines="8-23">
<description>
Cycle each solid odd-sided square to the next smaller centered square, wrapping a single pixel to a 9x9 square.

Input structure:
- The grid has one dominant background color.
- Every foreground object is a separated 4-connected monochrome solid square.
- Square side lengths are odd and belong to the cycle 1, 3, 5, 7, 9.

Rule:
- Clear the canvas to the background.
- Keep each square's color and center.
- Replace side length n by n - 2 when n > 1; replace side length 1 by side length 9.

Key insight:
- The visible operation is on square size, not color: side lengths cycle 1 -> 9 -> 7 -> 5 -> 3 -> 1.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="train" puzzle_id="8618d23e" lines="9-25">
<description>
Stagger the two horizontal input regions onto a larger maroon separator canvas.

Input structure:
- The input is a rectangular grid with an even number of rows.
- The top half and bottom half are the two regions to preserve.
- Maroon is absent from the input and is used only as the new separator color.

Rule:
- Create an output with one extra row and one extra column, filled with maroon.
- Copy the top half into the upper-left corner.
- Copy the bottom half one row below the top region and one column to the right.

Key insight:
The output is not a recoloring of either region; it inserts an L-shaped maroon
gap by diagonally offsetting the lower half from the upper half.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="train" puzzle_id="f1bcbc2c" lines="40-55">
<description>
Paint the top-connected middle channel between two orange path components teal.

Input structure
- The background is black.
- One foreground color forms exactly two 4-connected one-pixel-wide path components.
- Optional non-background control pixels may sit in the channel between the two paths.

Rule
- Dilate each path component by one cell in the 8-neighborhood and intersect the two dilations; this gives the one-cell-wide middle channel.
- A control pixel in that channel is a blocker only when its 4-neighbors touch fewer than both path components; otherwise it is part of the channel and is repainted.
- After removing blockers, paint the channel component connected to the top edge teal and keep all other pixels unchanged.

Key insight
- The desired pixels are not found by region filling; they are the geometric overlap of the two paths' one-cell halos, with weakly attached control pixels splitting the channel.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="eval" puzzle_id="291dc1e1" lines="7-22">
<description>
Reorder and stack rectangles by proximity to two colored edge lines.

Input structure:
- bg = most common color.
- Two edge lines (h + v) meet at a corner marked by a unique color.
- "Blue" line is color 1 if present, else the longer line; other is "red."
- Solid rectangles in the interior.

Rule:
- Sort rectangles by (distance to blue edge, distance to red edge).
- If the blue edge is vertical, rotate each rectangle 90° CCW.
- Stack top-to-bottom, centered horizontally.

Key insight: Edge-line colors encode a coordinate system; proximity to each axis determines stacking order and rotation.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="train" puzzle_id="7acdf6d3" lines="25-42">
<description>
Move the source-color pixels into the unique outline interior whose size matches the source count.

Input structure
- The grid has a dominant background and exactly two foreground colors.
- One foreground color forms 8-connected outline objects with row-wise interior cells.
- The other foreground color supplies a set of source pixels away from the chosen outline.
Rule
- For each possible outline color, compute each 8-connected outline object's
  row-span interior: background pixels strictly between the leftmost and
  rightmost outline pixels on each occupied row.
- Select the unique outline interior whose cell count equals the total
  number of pixels of the other foreground color.
- Erase the source pixels and paint that matched interior with the source color.
Key insight
- The colors are roles, not fixed IDs: the source color is identified by
  counting, and the destination is the outline with a matching interior area.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="train" puzzle_id="5e6bbc0b" lines="42-56">
<description>
Compact the checkerboard's moving color toward the edge marked by the teal pixel, leaving a maroon swept trail on the marker line.

Input structure
- A blue/black checkerboard with exactly one teal marker replacing a blue pixel on a non-corner edge.
- The marker's checkerboard parity identifies the moving color (blue); the opposite parity is background (black).

Rule
- If the marker is on the left or right edge, compact moving-color pixels in every row toward that side.
- If the marker is on the top or bottom edge, compact moving-color pixels in every column toward that side.
- Preserve the teal marker at its edge cell.
- On the marker's row or column, color cells swept by shifted moving pixels maroon unless the final compacted line occupies them.

Key insight
The marker selects both the compression direction and the checkerboard color class that moves; maroon records the movement trail only along that marked line.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="eval" puzzle_id="0934a4d8" lines="75-90">
<description>
Reveal the rectangular patch hidden by the solid marker using the grid's reflection symmetries.

Input structure:
- A square colored pattern with one solid rectangular marker color occluding a patch.
- The visible pattern has horizontal and vertical reflection symmetry on a large lower/right strip.
- The top and left outer strips also expose transpose counterparts for cells not recoverable by those reflections.

Rule:
- Find the unique color whose full extent is a solid rectangle; this rectangle is the output region.
- Infer the reflected strip start as the smallest row/column offset whose lower/right strips are mirror-consistent, ignoring marker pixels.
- For each marker pixel, copy the unique non-marker value in its horizontal/vertical reflection orbit; if that orbit is entirely hidden in the outer strip, use the transposed top/left-strip counterpart.

Key insight:
The marker is not part of the pattern; it covers a missing rectangle whose cells are still determined by the visible mirror or transpose copies elsewhere in the grid.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="eval" puzzle_id="88e364bc" lines="102-115">
<description>
Slide yellow pixels inside walled regions in the direction dictated by border-legend arrows.

Input structure:
- bg = most common color; yellow pixels sit inside enclosed regions bounded by colored walls.
- Border legends: small rectangles on the grid edge whose fill color C matches a region's border color. Each legend contains a center dot and an indicator dot encoding a direction.

Rule:
- For each enclosed region, identify its border color C and look up the legend with fill color C.
- The direction from the legend's center dot to its indicator dot gives the slide direction.
- All yellow pixels in that region slide in that direction, stacking against the far wall.

Key insight: Legend fill color links to region border color; the center→indicator offset encodes the movement vector.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="train" puzzle_id="3d588dc9" lines="95-109">
<description>
Replace the active black cutout's tail with background and mark the cutout side facing a colored source region.

Input structure:
- The dominant color is background, with black rectangle-plus-tail cutouts, one gray source region, and colored distractors.
- Non-gray colored regions and single pixels are distractors/blockers, even when they are multi-pixel staircase shapes.

Rule:
- Find each black cutout whose largest solid rectangle leaves exactly a two-pixel tail.
- The active gray source region is visible only by horizontal rays through background from a cutout's left or right side.
- Select the unique cutout side with the most same-source visible rows, erase that cutout's tail to background, and recolor the full selected vertical side pink (6).

Key insight:
- The colored regions do not move; their horizontal line-of-sight identifies which rectangular cutout is being opened and which side receives the pink mark.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="train" puzzle_id="e4888269" lines="67-88">
<description>
Recolor every target pixel by following the ordered two-column substitution legend.

Input structure
- A dominant background surrounds one multicolor vertical legend: a connected,
  fully filled two-column block with one color pair per row.
- Other non-background pixels or objects are the targets; divider or frame
  pixels are ordinary target-side structure that remain unchanged unless their
  color is rewritten by the legend.

Rule
- Read the legend rows from top to bottom as substitutions from the left
  column color to the right column color.
- For each non-background pixel outside the legend, scan those rows in order;
  whenever the current color equals the row's left color, replace it with the
  row's right color and keep scanning lower rows.
- Leave the legend itself and background pixels unchanged.

Key insight
- The same color may be rewritten multiple times, but only by substitutions
  that appear later in the vertical legend, so the row order is part of the
  symbol instruction.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="train" puzzle_id="fd02da9e" lines="26-41">
<description>
Replace each colored corner marker with its corner-specific inward stencil in the same color.

Input structure
- The grid is 8x8 with a single dominant background color.
- Foreground pixels are marker pixels, each located at one of the four corners.
- A marker's color is carried to the stencil selected by its corner position.

Rule
- Erase each corner marker back to the background.
- Top corner markers become an inward 2x2 block one cell away from the top and side edges.
- Bottom corner markers become the mirrored three-pixel inward bent stencil shown by the bottom-corner examples.

Key insight
- The marker position selects the fixed stencil; the marker color only supplies the paint color, so multiple corners apply independently.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="train" puzzle_id="337b420f" lines="41-55">
<description>
Keep the larger sprite from each separated panel and overlay those sprites into one 5x5 canvas.

Input structure:
- The input is a 5x17 grid: three 5x5 panels separated by one-cell black columns.
- Each panel has the same background color and exactly one foreground color split into two 4-connected components.

Rule:
- In each panel, discard the smaller foreground component and keep the unique largest component as the sprite.
- Process panels left to right, placing each sprite at its panel coordinates on a 5x5 background canvas.
- If a sprite would cover an already placed sprite, slide it by the unique nearest in-bounds offset that avoids overlap.

Key insight:
- The extra same-colored component in each panel is a distractor; the final answer is the non-overlapping overlay of the three largest components.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="train" puzzle_id="5751f35e" lines="29-43">
<description>
Complete sparse nested rectangular color layers into solid rectangles.

Input structure
The background is the dominant color, and every foreground color marks one
axis-aligned rectangular layer; the layer bboxes have unique areas and are
nested from outermost to innermost.
Rule
Take each color's bbox as its completed rectangle, except that a child bbox
which already matches at least three sides of its parent's one-pixel inset
expands to that full inset; repaint the rectangles largest to smallest.
Key insight
The scattered pixels are samples of nested rectangular regions, so the
output is a clean redraw of those regions with inner colors overwriting
outer colors.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="train" puzzle_id="320afe60" lines="41-56">
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

<reference_puzzle index="18" source="train" puzzle_id="58c02a16" lines="10-26">
<description>
Expand the top-left seed mask into a recursive colored pattern and tile it across the full grid.

Input structure
- The background is the dominant color.
- All foreground lies in a top-left rectangle whose last row and last column are a solid secondary color.
- Removing that secondary-color row and column leaves a seed region containing background plus one primary color.

Rule
- Use the primary-color pixels in the seed as a binary mask.
- Replace each primary seed cell with a copy of the seed mask in the primary color, and each background seed cell
  with a copy of the seed mask in the secondary color.
- Repeat and crop that expanded tile to the original grid shape.

Key insight
The solid L-shaped secondary-color border is a legend, not output content; it supplies the color used when the
seed's background cells are expanded.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="train" puzzle_id="e3fe1151" lines="36-55">
<description>
Fill the one blank in each corner region so all four regions have the same color multiset.

Input structure:
- The dominant background color forms one solid separator row and one solid
  separator column.
- The separator cross splits the grid into four equal rectangular corner regions.
- Each corner region contains exactly one background-colored blank pixel.

Rule:
- Count the non-background colors in each corner region.
- The completed region multiset is defined by the maximum observed count of
  each color across the four regions.
- Replace each region's blank pixel with the single color needed to match that
  completed multiset.

Key insight:
- The separator color inside a corner region is a missing value marker; comparing
  the four regions reveals the shared multiset without using numeric color order.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="train" puzzle_id="aaef0977" lines="25-41">
<description>
Replace an orange canvas with concentric Manhattan-distance color rings centered on the lone marker pixel.

Input structure:
- The grid is an orange background with exactly one non-orange marker pixel.
- The marker color is one of the colors in the fixed non-orange ring cycle.

Rule:
- The marker location is distance 0 and keeps the marker color.
- Each step of Manhattan distance away from the marker advances one position
  through the cycle green, yellow, black, grey, red, teal, maroon, pink, blue.
- Fill every pixel of the original grid by this distance-indexed cycle.

Key insight: The output is not copied from existing objects; it is the
distance field around the single marker, colored by a cyclic palette that
excludes the orange background.
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
