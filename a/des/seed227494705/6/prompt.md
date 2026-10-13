你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="bc93ec48" lines="48-63">
<description>
Copy each foreground object anchored in a true grid corner clockwise into the next corner.

Input structure
- The dominant color is the background.
- Foreground objects are monochromatic 4-connected components.
- Relevant objects are exactly those whose bounding box touches two adjacent borders at one grid corner.

Rule
- Start from the input unchanged.
- For each corner object, copy its pixel mask and color to the next corner clockwise: top-left to top-right, top-right to bottom-right, bottom-right to bottom-left, and bottom-left to top-left.
- Preserve each object's orientation by aligning its bounding box to the destination corner; stamps may overwrite existing pixels at the destination, but source pixels are not cleared.

Key insight
- Interior and side-only objects are distractors; only components actually anchored to a corner participate in the clockwise perimeter copy.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="train" puzzle_id="4e7e0eb9" lines="31-49">
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

<reference_puzzle index="3" source="train" puzzle_id="9f669b64" lines="26-48">
<description>
Push the central solid rectangle to the grid edge, splitting the perpendicular target rectangle around its path.

Input structure
- The background is the dominant color.
- There are exactly three monochrome foreground objects arranged on one
  horizontal or vertical axis.
- The middle object on that axis is a solid rectangular moving object.
- One end object is a solid rectangle or bar whose long side is perpendicular
  to the movement axis; this is the target to split.

Rule
- Move the middle rectangle along the axis toward the target until it is
  flush with the grid edge in that direction.
- Clear the original middle rectangle and the original target.
- Split the target through the moving rectangle's two-cell-wide path and
  shift the two halves one cell outward perpendicular to the movement.
- Keep the other end object unchanged.

Key insight
- The non-target end object acts as a pusher; the central rectangle travels
  away from it through the perpendicular target, opening a matching gap.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="train" puzzle_id="7e4d4f7c" lines="9-24">
<description>
Compress the L-shaped cue to three rows and add a pink copy of the top-row foreground mask.

Input structure
The grid has a dominant background color, a two-color top row, and all
foreground pixels below the top row lie in the first column as vertical markers.

Rule
Keep the input top row as output row 0 and the first marker row as output row 1.
Output row 2 is background except at the top-row foreground columns, which are
recolored pink.

Key insight
The lower repeated marker rows only identify the vertical cue; the generated
row is a recolored mask of the horizontal cue in the top row.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="train" puzzle_id="f8cc533f" lines="10-25">
<description>
Complete every colored sprite by applying the union of all aligned partial sprite masks.

Input structure:
- The dominant color is the background.
- Each non-background color appears as a separated partial copy of one common sprite mask.
- The copies share a top-left-aligned sprite window; some copies may be missing lower or right edge pixels.

Rule:
- Crop each same-color mask to its bounding box.
- Pad the crops to the maximum observed height and width, aligned at the top-left, and OR them into one complete template.
- Fill that template at every sprite location using the sprite's own color.

Key insight:
- No single input sprite is complete, but their missing cells are complementary, so the aligned union reveals the full mask.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="train" puzzle_id="6cbe9eb8" lines="79-94">
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

<reference_puzzle index="7" source="train" puzzle_id="34cfa167" lines="94-117">
<description>
Complete a four-corner framed rectangle from one decorated anchor rectangle and its diagonal mate.

Input structure:
- The background is the dominant color.
- Two same-color solid rectangular anchors of the same size sit on opposite
  diagonal corners, with the first anchor above-left of the second.
- The upper-left anchor has two one-cell-thick vertical control bars to its
  right and two one-cell-thick horizontal control bars below it.

Rule:
- Copy the anchor rectangle into the two missing corners of the implied
  larger rectangle.
- Repeat the right-side control bars horizontally between the left and right
  anchors, using their spacing as the period, and draw the outer top and
  bottom borders in the first right-control color.
- Repeat the below-side control bars vertically between the top and bottom
  anchors, using the same period, and draw the outer left and right borders
  in the first below-control color.

Key insight: The small control bars are the first two ticks of two periodic
frame patterns; the diagonal anchor supplies the opposite corner and the
missing two anchor copies.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="eval" puzzle_id="3dc255db" lines="37-53">
<description>
Eject content pixels from cup-shaped frames outward along each cup's tail direction.

Input structure:
- bg = background. Hollow cup/Y/funnel frames, each a unique border color.
- Each cup has a convergent tail spike on one side and an open mouth on the opposite side.
- Content pixels (a single non-border color per cup) sit inside the cup's mouth interior.

Rule:
- For each cup, identify its tail spike and tail direction.
- Erase all content pixels from the mouth interior.
- Draw a straight line of content-colored pixels from the spike outward in the tail direction,
  one pixel per erased content pixel; cells beyond the grid edge are dropped.

Key insight: The tail spike is the unique convergent point whose outward direction is opposite
the mouth opening where the content lives.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="train" puzzle_id="f3e14006" lines="21-39">
<description>
Build the woven rectangle determined by the two marked cells on each crossed bar.

Input structure
A black-background grid contains one long horizontal bar and one long vertical bar.
Each bar has a dominant base color and exactly two same-colored marker pixels.

Rule
The horizontal marker pair gives the output column span.  From the horizontal bar
row, walk toward the farther vertical marker; rows and columns alternate every
other cell.  Even step/even offset cells use the horizontal marker color until
the nearer vertical marker is reached, then use the vertical marker color.
Odd row steps use the horizontal base on marker columns and the original
crossing color on spacer columns.  Spacer columns on even row steps use the
vertical base, except at the switch row where they use the crossing color.

Key insight
The original cross cell records which base color wins at base/base lattice
crossings; the nearer vertical marker is the color-switch boundary.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="train" puzzle_id="9caf5b84" lines="28-45">
<description>
Recolor every pixel whose color is not one of the two most frequent input colors to orange.

Input structure:
- The grid contains at least three colors.
- Orange is absent from the input and is used only as the output replacement color.
- Exactly two colors occupy the top two frequency ranks; they may tie with each
  other, but no third color ties them at the cutoff.

Rule:
- Preserve every pixel whose color is one of the two most frequent input colors.
- Replace every other pixel with orange.

Key insight:
- The visible distinction is not the numeric identity of the colors but their
  population: the two majority colors remain, while all minority colors merge
  into the new orange color.
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

<reference_puzzle index="12" source="eval" puzzle_id="6e4f6532" lines="26-41">
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

<reference_puzzle index="13" source="train" puzzle_id="7e576d6e" lines="159-181">
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

<reference_puzzle index="14" source="train" puzzle_id="a644e277" lines="31-50">
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

<reference_puzzle index="15" source="train" puzzle_id="cf5fd0ad" lines="10-22">
<description>
Build a four-quadrant rotational tiling of the input square.

Input structure
- The input is a square grid treated as one complete sprite.

Rule
- Each output quadrant is a 2x2 repetition of a rotated copy of the whole input.
- The quadrant rotations are upper-left 180 degrees, upper-right 90 degrees
  clockwise, lower-left 90 degrees counterclockwise, and lower-right unchanged.

Key insight
- The output is a 4x4 grid of original-size blocks, not a per-pixel scale-up.
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

<reference_puzzle index="17" source="train" puzzle_id="b25e450b" lines="28-43">
<description>
Slide each black border rectangle to the opposite border, recoloring its swept stripe orange and leaving the rectangle black at the far edge.

Input structure
- The grid contains black rectangular 4-connected masks, each touching exactly one outer border.
- Grey cells form incidental pattern content; orange is the trail color used when a black mask slides.

Rule
- Treat all black rectangles as moving simultaneously.
- A rectangle touching the top or bottom edge moves vertically to the opposite edge in the same columns.
- A rectangle touching the left or right edge moves horizontally to the opposite edge in the same rows.
- Recolor the complete swept row or column stripe to orange, then stamp every destination rectangle in black.

Key insight
- Edge contact gives the slide direction; the output is the set of opposite-edge black rectangles plus their orange trails.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="eval" puzzle_id="2ba387bc" lines="6-21">
<description>
Sort scattered 4×4 squares into two columns by type: frames left, solids right.

Input structure:
- bg = most common color. Multiple 4×4 objects on bg.
- Frames: colored 1px border, bg-filled 2×2 interior.
- Solids: entirely filled with one non-bg color.

Rule:
- Classify each 4×4 object as frame or solid by its interior.
- Sort each group top-to-bottom by input row.
- Stack frames in left column (cols 0–3), solids in right (cols 4–7).
- Output is 8 wide; height = 4 × max(#frames, #solids), bg-padded.

Key insight: Frame vs solid is determined solely by whether the 2×2 interior matches bg.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="eval" puzzle_id="f931b4a8" lines="10-27">
<description>
Count the two upper quadrant control regions to size an output made from the lower quadrant pattern.

Input structure:
- The grid is an even square split into four equal quadrants.
- Black is the background.
- The upper-left and upper-right quadrants are control regions.
- The lower-left quadrant is a color source tile; the lower-right quadrant is either a stencil tile or empty.

Rule:
- Output height is the number of non-background pixels in the upper-left quadrant.
- Output width is the number of non-background pixels in the upper-right quadrant.
- If the lower-right stencil has foreground pixels, tile it over the output; each stencil background pixel is filled from the lower-left quadrant cell at the corresponding coarse tile position.
- If the lower-right stencil is empty, tile the lower-left quadrant directly over the output.

Key insight:
- The top quadrants specify only dimensions, while the bottom quadrants specify a two-level repeated pattern: a fine stencil and coarse fill colors.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="eval" puzzle_id="4c3d4a41" lines="15-28">
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
   - **禁止使用**（这些是算法量而非视觉量，会迫使解题者依赖公式而非视觉直觉）：曼哈顿 / 欧氏 / Chebyshev 距离作为精确查表键、凸包面积、二进制位编码 / 位运算、模运算 / 余数、拓扑距离（BFS / 最短路 / 跳数）、精确几何质心坐标、凸角 / 凹角计数等复杂且不易被看出来的计算都是禁止使用的。
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

