你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="1190bc91" lines="36-49">
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

<reference_puzzle index="2" source="train" puzzle_id="db615bd4" lines="111-130">
<description>
Thicken the sparse frame and stamp the external sparse sprites into it in positional order.

Input structure
The grid has a dominant background color, a regular lattice color, one large
sparse rectangular frame, and one or more external monochrome sprites whose
pixels also sit on a two-step lattice.

Rule
Use the largest sparse rectangular perimeter as the frame. Clear that frame
region and the external sprite pixels to background, draw a solid one-pixel
border in the frame color, solidify each sparse sprite into its contiguous
two-step mask, and place the sprites inside the frame with equal gaps.
Wide frames receive sprites left-to-right by source position; tall frames
receive them top-to-bottom by source position.

Key insight
The frame supplies ordered slots: the sprite-to-slot assignment is by visible
source order, not by numeric color or by matching individual cells.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="train" puzzle_id="e4888269" lines="67-88">
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

<reference_puzzle index="4" source="train" puzzle_id="412b6263" lines="39-54">
<description>
Infer the background, rotate the whole input 90 degrees counterclockwise, and stack two framed copies vertically.

Input structure:
- Blue is absent from the input.
- One rare marker color appears in exactly two pixels.
- The background is the unique non-marker color whose removal leaves exactly two 4-connected foreground objects, each with one marker pixel and one body color.
- The whole grid, including background and foreground objects, is the sprite to transform.

Rule:
- Rotate the entire input grid 90 degrees counterclockwise.
- Create a blue one-pixel frame around two vertical copies of the rotated grid.
- Use the input background color for the two endpoints of every horizontal frame/separator row.

Key insight: The output is not a crop of the foreground objects; it is two copies of the complete rotated input, separated and enclosed by a newly added blue border.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="train" puzzle_id="b745798f" lines="18-33">
<description>
Expand each 3-pixel L marker into a same-color corner L on an empty background.

Input structure
- The grid is an odd square with one dominant background color.
- Every foreground object is a monochromatic 4-connected 3-pixel L inside a
  2x2 box.

Rule
- Clear the grid to the background.
- For each marker, map its L orientation to the corresponding outer corner
  and draw that corner L in the marker color with arm length half the grid.

Key insight
- The small marker is a miniature corner; its orientation, not its position,
  chooses which outer corner receives the expanded L.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="eval" puzzle_id="c4d067a0" lines="9-23">
<description>
Complete a grid of colored blocks to match a minimap legend in the top-left corner.

Input structure:
- bg = most common color.
- Top-left corner: a small sprite of single colored pixels forming a minimap pattern on a regular lattice.
- Main area: uniformly colored square blocks (NxN) arranged on a regular grid; some positions are empty (bg).

Rule:
- The legend pixels define a color pattern; each pixel's lattice position maps to a block position in the main grid via a fixed scale factor and offset.
- Existing colored blocks confirm the scale/offset alignment.
- For every legend pixel whose corresponding grid position is empty, place a new block of that color.

Key insight: The top-left pixel pattern is a scaled-down minimap of the block layout; the mapping is inferred from blocks already present.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="train" puzzle_id="c4d1a9ae" lines="28-48">
<description>
Fill each vertical foreground band's background cells with the color of the next band to its right.

Input structure
- The dominant color is the background.
- Each other color occupies a separate vertical band defined by that color's
  minimum and maximum columns.
- Within a band, all cells are either that band's color or background.

Rule
- Order the bands from left to right.
- In each band, replace background pixels with the color of the next band in
  that order, wrapping from the rightmost band back to the leftmost.
- Preserve all existing foreground pixels and all background pixels outside
  the bands.

Key insight
- The column bands form a cyclic palette legend: a band's gaps are completed
  by borrowing the color from its right-hand neighbor, not by changing the
  band's existing pixels.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="train" puzzle_id="e734a0e8" lines="44-59">
<description>
Copy the unique colored sprite tile into every tiled cell marked by an internal separator-colored pixel.

Input structure
- Full same-color separator rows and columns partition the grid into equal rectangular tiles.
- One tile contains the only non-background, non-separator colored sprite.
- Target tiles are otherwise background-colored but contain at least one internal pixel of the separator color.

Rule
- Keep the separator lines and the source sprite tile unchanged.
- Replace each marked target tile with an exact copy of the source tile.
- Unmarked background-only tiles remain unchanged.

Key insight
- The internal separator-colored pixels are placement markers; the single colored tile is the template to stamp into those marked cells.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="train" puzzle_id="342ae2ed" lines="34-49">
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

<reference_puzzle index="10" source="train" puzzle_id="e3f79277" lines="8-23">
<description>
Replace a corner L with a same-color right-triangle outline in the corresponding corner of an enlarged canvas.

Input structure:
- The input is a square grid with one background color and one foreground color.
- The foreground is a one-pixel-thick L inside a square bbox.
- The L touches exactly one horizontal grid edge and one vertical grid edge, identifying a corner.

Rule:
- Fill a square output canvas of side 3n-2, where n is the input side length, with the background color.
- Double the L leg length to get the triangle side length.
- Draw the two perpendicular triangle legs on the same output corner edges and connect their free endpoints with the diagonal.

Key insight:
- The input L encodes the corner, color, and side length of the triangle outline; the missing side is the diagonal between the L's two free endpoints.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="eval" puzzle_id="64efde09" lines="68-83">
<description>
Copy colored control-ray ports from one rectangular sprite orientation to every matching transformed sprite.

Input structure:
- bg is the most common color.
- Several dense 2-by-N or N-by-2 rectangular sprites are dihedral transforms of one template.
- Isolated single-pixel control markers sit on rays projected outward from the dominant long edge of some sprites.

Rule:
- For each sprite, the active edge is the long side whose cells have the strongest same-color majority.
- A control marker seen first along an outward ray from an active-edge cell assigns its color to that template cell.
- On every transformed sprite, draw that assigned color outward from the corresponding active-edge cell through bg cells until blocked or out of bounds.

Key insight:
- The marker positions are not absolute; they label cells of a shared rectangular template, then the labels follow each sprite's rotation/reflection.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="train" puzzle_id="3d6c6e23" lines="10-27">
<description>
Move each vertical stream of pixels into a bottom-aligned centered pile.

Input structure
- A mostly background grid contains one or more vertical source columns of
  foreground pixels.
- Each source column has n^2 foreground pixels, and the top-to-bottom color
  sequence splits into monochrome groups of lengths 1, 3, 5, ...

Rule
- Clear the canvas to background.
- For each source column, read the foreground colors from top to bottom and
  rebuild them at the bottom below the same column as odd-width centered
  rows of lengths 1, 3, 5, ...

Key insight
- The scattered vertical pixels are falling grains; a square number of
  grains decomposes into the odd row widths of a centered pyramid.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="eval" puzzle_id="a6f40cea" lines="74-88">
<description>
Complete partial paths inside a framed rectangle by projecting external color traces inward.

Input structure:
- bg, a 1px rectangular frame (border color) with uniform interior fill.
- Colored components outside the frame touching its edges.

Rule:
- Output = the frame's interior, filled with the fill color.
- Map each outside component's frame-touch points into interior coordinates.
- Monochrome components become solid Manhattan stripes between mapped endpoints.
- Two-color components become alternating-color Manhattan paths.

Key insight: Endpoint colors of two-color traces anchor the alternating phase deterministically.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="train" puzzle_id="6cbe9eb8" lines="79-94">
<description>
Rebuild the foreground rectangles as a largest-to-smallest lower-left nesting.

Input structure
- A periodic wallpaper background is overlaid by foreground colors that each form a rectangular frame or solid rectangle.
- The largest foreground rectangle is the output frame; smaller foreground rectangles are the objects to nest inside it.
- A frame may be partially hidden by a smaller rectangle, but its visible pixels still determine its bbox.

Rule
- Ignore the wallpaper and crop the result to the largest foreground rectangle's bbox size.
- Sort foreground rectangles by bbox area from largest to smallest.
- Reconstruct each frame or solid rectangle, placing every next smaller rectangle in the lower-left of the current rectangle's usable region.

Key insight
- The scattered background colors are not objects; the meaningful order is the unique descending rectangle size chain, and each step nests into the previous rectangle's lower-left interior.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="train" puzzle_id="1b59e163" lines="54-69">
<description>
Copy each marked sprite to every standalone marker of the same color and erase the original sources.

Input structure:
- The background is the dominant color.
- Each source is an 8-connected component containing a monochrome sprite and one marker pixel of another color.
- Other isolated marker pixels may be destinations; marker colors with no source sprite are distractors.

Rule:
- Clear the grid to background.
- For every source marker color, place the sprite at each other marker of that color using the source marker as the anchor.
- Keep each destination marker pixel in place; do not keep source sprites, source markers, or distractor markers.

Key insight:
- A colored marker is not part of the sprite geometry; it is the anchor that says where same-colored copies of that sprite should be stamped.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="train" puzzle_id="6bcdb01e" lines="10-24">
<description>
Extend the edge-touching green start bar through the corridor, turning at each teal blocker until the path leaves the grid.

Input structure:
- The background is the most common color.
- Exactly two foreground colors are present: a 2-pixel 4-connected start bar and many blocker pixels.
- Exactly one pixel of the start bar lies on the grid border.

Rule:
- Start from the border pixel of the bar and move through the other bar pixel into the grid.
- Paint every traversed non-blocker pixel with the start color.
- When the next cell is a blocker, turn to the unique perpendicular non-blocker neighbor; when the next cell is outside the grid, stop.

Key insight: The blocker pixels define a one-cell-wide corridor, so every interior collision has exactly one valid 90-degree turn.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="train" puzzle_id="3ad05f52" lines="128-142">
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

<reference_puzzle index="18" source="train" puzzle_id="3cd86f4f" lines="6-18">
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

<reference_puzzle index="19" source="train" puzzle_id="fe45cba4" lines="72-89">
<description>
Move the left-edge fragment into its same-colored anchor fragment to complete a solid rectangle.

Input structure
- The grid has a dominant background color.
- Exactly one foreground color is split into two 4-connected objects: one
  touches the left edge, and the other is an anchor away from the left edge.
- The translated left-edge object and the anchor can form one filled rectangle.
Rule
- Find the unique split color whose left-edge object can be translated onto
  background cells beside the other same-colored object so their union is a
  solid axis-aligned rectangle.
- Clear the left-edge object's original cells to the background.
- Paint that object at the unique completing offset, preserving all other cells.
Key insight
- The active color is identified by object structure and rectangle completion,
  not by a fixed color id; the same-colored anchor object stays in place.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="train" puzzle_id="18286ef8" lines="17-31">
<description>
Use the pink control pixel to choose where the maroon pixel belongs inside the grey 3x3 region, and turn the control pixel maroon.

Input structure:
- The grid contains one 3x3 region made from eight grey pixels and one maroon pixel.
- There is exactly one pink control pixel outside that region.

Rule:
- Restore the old maroon cell in the 3x3 region to grey.
- Compare the pink pixel to the 3x3 region: above/same band/below selects top/middle/bottom, and left/same band/right selects left/middle/right.
- Paint the selected cell maroon and recolor the pink control pixel maroon.

Key insight:
- The pink pixel acts as a relative-position marker for a 3x3 target inside the grey region.
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
