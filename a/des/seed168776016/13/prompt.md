你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="1d61978c" lines="8-22">
<description>
Recolor the two diagonal foreground-bar orientations, assigning teal to the larger orientation and red to the smaller one.

Input structure
- The grid has one dominant background color and one foreground color.
- Foreground pixels form one-cell-thick diagonal bars.
- Every foreground pixel belongs to exactly one diagonal orientation.

Rule
- Keep the background unchanged.
- Classify each foreground pixel by whether it has a diagonal neighbor on the NW/SE axis or the NE/SW axis.
- Recolor the orientation with more foreground pixels teal and the other orientation red.

Key insight
- The output colors depend on the relative population of the two diagonal orientations, not on a fixed slash direction or position.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="train" puzzle_id="c9680e90" lines="52-67">
<description>
Move each mirrored bottom source pixel to the far end of its trail and redraw the top half as its reflection.

Input structure
- A solid non-background row separates equal-height top and bottom halves.
- The top half has one marker color on the background.
- In the bottom half, exactly one color has positions that mirror the top markers across the separator, and the other color forms trails touching those source pixels.

Rule
- Identify the bottom source color by reflecting each bottom color into the top half and matching the top marker set.
- For every bottom source pixel, follow its unique 4-connected trail of the other bottom color to the trail endpoint.
- Clear the old top markers, bottom sources, and trails; place the source color at each endpoint and place the top marker color at the reflected endpoint.

Key insight
- The top half is only a mirror display of the bottom moving sources; the actual motion is encoded by the trail attached to each bottom source pixel.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="eval" puzzle_id="d59b0160" lines="39-54">
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

<reference_puzzle index="4" source="train" puzzle_id="8a6d367c" lines="52-67">
<description>
Complete the framed area by expanding the unique upper template that contains the existing seed pixels.

Input structure:
- A teal background contains small colored template masks above a larger rectangular frame.
- The frame encloses one fill color arranged as a partial block-scaled copy of one template.
- The frame interior dimensions are an integer square scale of the matching template bbox.

Rule:
- Crop to the rectangular frame.
- Identify the unique upper template whose block-scaled mask covers every existing fill pixel inside the frame.
- Reset the frame interior to background and paint the whole scaled mask with the fill color.

Key insight:
The lower colored pixels are seed cells that reveal which upper template should be completed inside the frame.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="train" puzzle_id="dc46ea44" lines="11-25">
<description>
Move the lower sprite and its pink guide into the upper panel, aligning the sprite to the guide's lower-left branch.

Input structure
- The grid has one dominant background color and one full-width separator row.
- Below the separator are exactly two foreground roles: a five-pixel guide marker and one monochrome sprite.
- The guide marker has normalized shape {(0,2), (1,2), (2,0), (2,2), (3,1)} and starts immediately below the separator.

Rule
- Clear the canvas to the background, preserving the separator row.
- Translate the guide marker upward so its top row becomes row 0.
- Move the sprite, preserving its color and shape, so the sprite bbox bottom-right aligns to the translated guide's lower-left branch.

Key insight
- The marker supplies the destination anchor: its lower-left branch cell is where the moved sprite's bbox bottom-right must land.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="train" puzzle_id="e4888269" lines="67-88">
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

<reference_puzzle index="7" source="eval" puzzle_id="a32d8b75" lines="262-275">
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

<reference_puzzle index="8" source="eval" puzzle_id="136b0064" lines="25-39">
<description>
Draw a polyline by decoding 3x3 glyph instructions read across a divided grid.

Input structure:
- Left of a full-height vertical divider: 3x3 colored glyphs on bg.
- Right of divider: a single gray (5) start pixel.

Rule:
- Remove the divider; shift the gray start into the left canvas.
- Read glyphs in column-major order (left→right, top→bottom).
- Each glyph's shape (pixel area) encodes a segment: area 4 → down 2, area 5 → right 3, area 6 → left 4, area 7 → left 2.
- Append segments sequentially, stepping down one cell between each.

Key insight: Glyph pixel count (area) uniquely determines both segment length and direction — the shape itself is the instruction.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="train" puzzle_id="973e499e" lines="7-22">
<description>
Expand each input pixel into a miniature same-color mask of the whole grid.

Input structure
- The input is a non-empty two-dimensional grid of foreground colors.
- Black is reserved for the empty/background pixels introduced in the expanded output.
- The output has one input-shaped block for each input pixel.

Rule
- For every input pixel with color c, replace that pixel by a block the same shape as the input.
- Inside the block, write c exactly where the original input also has color c.
- Write black everywhere else in the block.

Key insight
Each pixel selects the global mask of its own color; the result is the blockwise self-mask of the grid.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="eval" puzzle_id="4a21e3da" lines="8-20">
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

<reference_puzzle index="11" source="train" puzzle_id="a644e277" lines="31-50">
<description>
Extract the lattice subregion bounded by the four background-colored corner holes.

Input structure:
- A regular one-cell-thick lattice has one line color, one background color,
  and sparse marker pixels.
- Exactly four lattice-line intersections are background-colored holes instead
  of line-color pixels.

Rule:
- Find the unique non-background color that forms regular horizontal and
  vertical lattice lines.
- Use the four background-colored lattice intersections as the corners of one
  rectangle.
- Return the inclusive crop between those corner rows and columns.

Key insight:
- The crop is marked by the missing lattice intersections, not by the marker
  pixels inside the cells.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="eval" puzzle_id="7b5033c1" lines="7-21">
<description>
Straighten a colored snake path into a single-column strip preserving pixel order.

Input structure:
- bg = most common color.
- Non-bg pixels form a single 4-connected path (snake) with exactly two endpoints.
- Each path pixel may be a different non-bg color.

Rule:
- Find the two endpoints (cells with only one path neighbor) of the snake.
- Begin at the top-left-most endpoint and trace the path cell by cell.
- Output an (N×1) column listing each cell's color in traversal order.

Key insight: The path is a simple chain with no branches; choosing the top-left-most endpoint fixes a unique reading direction.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="train" puzzle_id="8618d23e" lines="9-25">
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

<reference_puzzle index="14" source="train" puzzle_id="25e02866" lines="14-31">
<description>
Merge same-sized filled rectangular objects by overlaying their internal marker pixels.

Input structure:
- A dominant background surrounds two or more separated 4-connected foreground objects.
- Every object is a filled rectangle of the same size.
- Each rectangle has the same majority base color plus one or more non-base marker pixels.

Rule:
- Create one output rectangle with the shared object size, filled with the shared base color.
- For each input rectangle, copy every non-base marker pixel to the same relative location
  in the output.
- Marker overlays must either land on base color or agree in color with an existing marker.

Key insight:
The separated rectangles are layers of the same template; their colored marker pixels
combine by local coordinates while the repeated base fill is just the output canvas.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="eval" puzzle_id="58490d8a" lines="9-21">
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

<reference_puzzle index="16" source="train" puzzle_id="a834deea" lines="20-39">
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

<reference_puzzle index="17" source="train" puzzle_id="d753a70b" lines="81-93">
<description>
Red diamond outlines contract by one layer and gray diamond outlines expand by one layer while other objects stay fixed.

Input structure
- A 16x16 orange-background grid contains one-cell-thick Manhattan-diamond outlines.
- Red and gray outlines are the changing objects; other colored diamond outlines are static.
- Some outlines may be clipped by the grid boundary, but their visible pixels still fit a clipped Manhattan-distance contour.
Rule
- Clear all red and gray pixels.
- Red objects are redrawn with the same diamond center and radius minus one.
- Gray objects are redrawn with radius plus one; an ambiguous edge-clipped growing arc continues away from the nearest perpendicular edge.
Key insight
- The task is not about moving whole objects: each changing object is a contour of a diamond, and the visible operation is one step of contour contraction or expansion.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="train" puzzle_id="b74ca5d1" lines="157-172">
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

<reference_puzzle index="19" source="train" puzzle_id="396d80d7" lines="18-30">
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

<reference_puzzle index="20" source="eval" puzzle_id="a251c730" lines="8-21">
<description>
Stamp source sprites into a destination frame via color-matched singleton markers.

Input structure:
- Two bordered rectangles: source (multi-cell sprites) and destination (singleton markers).
- Each sprite has an anchor pixel — its least-frequent color. Markers share these colors.

Rule:
- Match each destination marker to the source sprite with the same anchor color.
- Translate the sprite so its anchor overlaps the marker position.
- Output is the destination rectangle, interior cleared before stamping.

Key insight: The rarest color per sprite doubles as a template key and positioning anchor.
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

请一次性生成5个`<new_puzzle>`，若有较多合适的创新，也可以生成更多，不设上限，但不要输出额外解释。

注意事项：
1. 任务不应依赖大量先验知识或复杂计算，而应依赖对模式、结构与空间关系的直觉推理。
2. 关于"计算"的边界：
   - **允许使用**（这些都是人类能心算的视觉量）：数像素 / 孔洞 / 对象 / 邻居（≤10 量级）、比大小、按某个可见属性排序、奇偶判断、取最大 / 最小、用一个计数值作为另一个等量几何量（如 N 个像素 → 长度 N 的射线、面积 N → 边长 N）、判断接触 / 相邻 / 包含、粗略中心位置（簇的视觉中点）、"距离"作为"几步"而不是公式等人类能够直观感受计算的内容。
   - **禁止使用**（这些是算法量而非视觉量，会迫使解题者依赖公式而非视觉直觉）：曼哈顿 / 欧氏 / Chebyshev 距离作为精确查表键、凸包面积、二进制位编码 / 位运算、模运算（取模） / 余数、拓扑距离（BFS / 最短路 / 跳数）、精确几何质心坐标、凸角 / 凹角计数等复杂且不易被看出来的计算都是禁止使用的。
3. 问题必须能从少量示例中推断出来（通常为 3 对训练样例 + 1 个测试样例）。设计时从人类解题者的视觉角度思考：规则涉及的所有关键要素（对象、关系、属性）必须在网格上肉眼可见，不应依赖人眼无法直接观察到的隐含信息（例如颜色id和颜色亮度是禁止使用的）或需要推算的隐藏属性。规则背后的逻辑本身可以有一定挑战性，但解题所需的所有线索必须能从网格中直接获取。
4. 生成的问题描述旨在促进后续用于创建题目网格的代码生成。因此，对输入 / 输出网格的描述以及规则总结必须极其详细；语言风格应易于理解；对于特定对象（如箭头），必须在描述中明确其绘制方式。
5. 描述谜题时避免使用 0 或 1 这类颜色编号；应使用颜色名称，例如 "red"。可用颜色如下：Black、Blue、Red、Green、Yellow、Grey、Magenta、Orange、Azure、Maroon。
6. 新题的核心推理机制应明显不同——即从观察样例到归纳规则的认知路径不同，而不仅仅是场景、对象或颜色不同。
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

