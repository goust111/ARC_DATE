你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="fe45cba4" lines="72-89">
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

<reference_puzzle index="2" source="train" puzzle_id="6ca952ad" lines="15-30">
<description>
Drop every large foreground object straight down to the bottom while leaving small marker objects fixed.

Input structure:
- The grid has a dominant background color and exactly one foreground color.
- The foreground is split into separated 4-connected objects.
- Objects of area at most three are stationary markers; larger objects are moving sprites.

Rule:
- Clear each moving sprite from its original position.
- Translate each moving sprite vertically downward until its bbox bottom reaches the grid bottom.
- Repaint each translated sprite in its original color; small markers stay unchanged.

Key insight:
- Object area separates the active falling sprites from the small fixed markers, and all active sprites follow the same vertical path.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="eval" puzzle_id="de809cff" lines="86-103">
<description>
Turn black dents in two colored regions into marked opposite-color patches.

Input structure:
- Black background with two large foreground colors.
- The large regions contain small black dents and a few one-cell color specks
  or protrusions along region boundaries.

Rule:
- Locally clean the two foreground regions: isolated specks become black, and
  one-cell protrusions surrounded by the other region take that region's color.
- Each original black dent whose cleaned 3x3 neighborhood is mostly one
  foreground color becomes the center of a 3x3 patch in the opposite color.
- The dent center itself is recolored teal (8).

Key insight: The teal pixels mark the original black defects; the surrounding
3x3 patch uses the other foreground color, after first removing boundary noise.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="train" puzzle_id="9f5f939b" lines="42-57">
<description>
Mark the missing center of each plus-shaped arrangement of four two-pixel bars with yellow.

Input structure:
- The background is the dominant color.
- The single foreground color forms isolated 4-connected objects, each a horizontal or vertical two-pixel bar.
- Some bars form an implied plus sign: two horizontal bars and two vertical bars around an empty center, with a one-pixel gap between the center and each bar.

Rule:
- Keep the input unchanged except at implied plus-sign centers.
- For every background pixel with foreground bars at offsets 2-3 cells left, right, above, and below, recolor that center yellow.
- The one-pixel gaps immediately next to the center in the four cardinal directions must remain background.

Key insight:
- The yellow pixels are not extending a line; they identify the shared missing center where four separated two-pixel bars align as a plus sign.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="train" puzzle_id="458e3a53" lines="19-32">
<description>
Extract the compact table of solid cells hidden inside a separator lattice.

Input structure
- The grid has full-width and full-height bars of one separator color, dividing it into rectangular cells.
- Most cells contain a repeating multicolor pattern, while a contiguous rectangle of cells is solid monochrome.

Rule
- Split the input into lattice cells using the full separator rows and columns.
- Keep the cells whose full interior is one color, crop to their occupied rectangle, and output those colors.

Key insight
- The periodic pattern is noise; only completely solid lattice cells encode the reduced output grid.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="eval" puzzle_id="800d221b" lines="7-22">
<description>
Recolor a path-network by adjacent seed regions.

Input structure:
- bg = most common color.
- Path color P forms a thin orthogonal network with a solid square hub (NxN).
- Seed colors appear in colored regions (non-bg, non-P) adjacent to path branches.

Rule:
- Infer P as the non-bg color whose largest component is most network-like.
- Find the largest solid k×k all-P square (the hub); exclude it from recoloring.
- Recolor each path branch to the seed-region color it touches most.
- Recolor the hub center pixel to the seed whose branches touch the hub most.

Key insight: The hub stays uncolored; branches inherit color from adjacent seed regions by adjacency count.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="train" puzzle_id="f1bcbc2c" lines="40-55">
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

<reference_puzzle index="8" source="train" puzzle_id="8e301a54" lines="8-23">
<description>
Move every foreground object straight downward by a number of rows equal to its own pixel count.

Input structure:
- The grid has one dominant background color.
- The foreground is made of separated 4-connected monochromatic objects.
- Objects may move partly below the bottom edge of the grid.

Rule:
- Clear the canvas to the background.
- For each foreground object, compute its area in pixels.
- Redraw the object at the same columns and that many rows lower, clipping pixels below the grid.

Key insight:
- The downward displacement is the object's own area; color and shape only identify which pixels move together.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="train" puzzle_id="878187ab" lines="10-24">
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

<reference_puzzle index="10" source="eval" puzzle_id="b9e38dc0" lines="7-18">
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

<reference_puzzle index="11" source="train" puzzle_id="342dd610" lines="18-36">
<description>
Move every foreground pixel by the fixed offset indicated by its color.

Input structure:
- The grid has a unique dominant background color.
- The foreground is a sparse set of colored pixels.
- Each foreground color is one of the trained motion colors: orange, red,
  maroon, or blue.

Rule:
- Clear the canvas to the background color.
- Translate orange pixels up 2 cells, red pixels left 2 cells, maroon
  pixels down 2 cells, and blue pixels right 1 cell.
- Preserve each moved pixel's original color.

Key insight:
The foreground color is the motion instruction; position only determines
where that color-specific displacement lands.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="train" puzzle_id="996ec1f3" lines="16-31">
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

<reference_puzzle index="13" source="train" puzzle_id="ac0c2ac3" lines="7-19">
<description>
Fill each centered square ring with the color of the marker at that ring distance.

Input structure
The input is an odd square grid with a dominant background color and one
non-background marker for every Chebyshev distance from the grid center.
Rule
The output replaces the background with concentric square rings centered on
the middle cell; every cell at distance d from the center takes the color of
the unique marker whose distance from the center is d.
Key insight
The scattered markers are not objects to move; their Chebyshev distances
form a palette for the centered square rings.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="eval" puzzle_id="6ffbe589" lines="83-98">
<description>
Crop a board cluster and rotate its color layers based on outside cue sprites.

Input structure:
- bg = most common color.
- A main multi-color board cluster (largest foreground region).
- Small monochrome cue sprites outside the board for the colors that rotate.

Rule:
- Tight-crop around the board cluster.
- For each color layer inside the crop, find its same-color cue outside.
- Rotate that layer clockwise by (cue pixel count mod 4) × 90°.
- No outside cue → no rotation.

Key insight: Cue sprite size mod 4 encodes rotation amount for its matching color layer.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="eval" puzzle_id="38007db0" lines="9-22">
<description>
Extract the unique cell from each lattice row into a single-column lattice.

Input structure:
- Grid divided into equal square cells by uniform gridlines of color L.
- Cell interiors filled with bg; each cell has an inner sprite pattern.
- Each row of cells has exactly one cell with a unique pattern; the rest are copies.

Rule:
- For each lattice row, find the cell whose inner pattern appears exactly once (the "odd one out").
- Construct a single-column lattice preserving the original row heights, containing only the unique cells.

Key insight: The task is per-row "odd one out" — every cell in a row is identical except one oddball, which must be extracted.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="eval" puzzle_id="89565ca0" lines="120-136">
<description>
Convert each colored partitioned frame into a horizontal bar whose length is its number of enclosed regions.

Input structure:
- Black background with several axis-aligned colored line drawings.
- The line drawings are damaged by scattered pixels of one non-background noise color and by overlaps with other drawings.
- Each real drawing is a rectangular partitioned frame enclosing one or more background regions.

Rule:
- For each non-background color, reconstruct its horizontal and vertical wall segments from dense row/column evidence.
- Count the 4-connected background regions fully enclosed by those reconstructed walls.
- The color with no enclosed region is the noise/pad color.
- Sort drawing colors by enclosed-region count; equal counts are ordered by which reconstructed frame covers the other.

Key insight:
- The scattered color is not an object; it only pads the histogram, while each true object's room count is stable after rectilinear wall reconstruction.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="train" puzzle_id="833966f4" lines="8-22">
<description>
Swap the upper two-pixel segment and the lower two-pixel segment around the fixed center pixel.

Input structure:
- The grid is a 5x1 vertical column of individual colored pixels.
- The middle pixel is a separator between two same-size two-pixel segments.

Rule:
- Keep the middle pixel in place.
- Reverse the order of the two pixels above the middle.
- Reverse the order of the two pixels below the middle.

Key insight:
- The transformation is local vertical order reversal on each side of the center, not a color-specific rule.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="train" puzzle_id="d749d46f" lines="32-51">
<description>
Pack each solid foreground rectangle twice: wide across the top and tall from the bottom.

Input structure
The grid has one background color and one foreground color; the foreground forms
separate solid rectangles, ordered from left to right.

Rule
Build a 10-row background canvas whose width is the packed width of the wide
versions of the rectangles. For each source rectangle, paint a top-aligned
rectangle of shape min(h,w) by max(h,w), with one background column between
consecutive rectangles. Then, starting again from the left edge, paint a
bottom-aligned rectangle of shape max(h,w) by min(h,w) for each source rectangle,
again separated by one background column.

Key insight
The input rectangles are not copied in their original orientations; each object
contributes one horizontal-facing and one vertical-facing normalized rectangle,
preserving left-to-right object order.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="eval" puzzle_id="6e4f6532" lines="26-41">
<description>
Dock train sprites onto station groups by matching coupler count, then orient via divider-color markers.

Input structure:
- bg; colored stripes (edge + internal) partition grid into rectangular regions.
- Color 8 forms train bodies; color 9 marks both isolated stations and train-attached couplers.
- Stripe-colored pixels on trains are orientation markers.

Rule:
- Stations = 8-connected 9-clusters not adjacent to any 8. Trains = components containing 8.
- Match train to station group where |couplers| == |station size|.
- Rotate/flip train so marker pixels face the same-color boundary stripe of the target region.
- Translate so couplers align exactly onto station cells; erase original position.

Key insight: Coupler count selects the docking target; marker color encodes which boundary stripe to face, fixing the unique orientation.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="train" puzzle_id="7e576d6e" lines="159-181">
<description>
Draw the marker-colored route through the centers of all doors in the parallel wall maze.

Input structure:
- The grid has a background, parallel full-span walls, one three-pixel door
  segment in each wall, and two marker pixels sharing a color.
- Walls are either all vertical or all horizontal, and the markers are on
  opposite outer sides of the wall set.

Rule:
- Keep the scene unchanged except for the route.
- Visit the door centers in wall order from the upper/left marker side to
  the lower/right marker side.
- In each corridor, draw an orthogonal path to the cell just before the next
  wall, move along that wall to the door center, cross the center, and
  continue to the next corridor; after the last door, connect to the other
  marker.

Key insight:
- The three-pixel door segments mark the only crossing points through a set
  of parallel walls, so the output is the corridor path threaded through
  those centers.
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
