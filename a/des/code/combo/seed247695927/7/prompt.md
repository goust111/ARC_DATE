你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="4e7e0eb9" lines="31-49">
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

<reference_puzzle index="2" source="train" puzzle_id="aaecdb9a" lines="10-26">
<description>
Convert scattered foreground objects into bottom-aligned count bars in a fixed five-color order.

Input structure
The grid has a dominant background color and foreground objects in the task
palette: grey, red, teal, maroon, and pink. Objects are monochromatic
8-connected components; some palette colors may be absent.

Rule
For each palette color in order grey, red, teal, maroon, pink, count its
8-connected objects. Create a background grid with five columns and height
equal to the largest count, then fill each column upward from the bottom
with its palette color for that count.

Key insight
The output does not preserve object shapes or positions; it is a histogram
of per-color 8-connected object counts, using the learned palette order.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="eval" puzzle_id="e3721c99" lines="69-83">
<description>
Recolor grey shapes by matching their hole count to a color legend.

Input structure:
- bg = most common color; grey (5) sprites scattered across the grid.
- Colored legend shapes (non-bg, non-grey, at least 2×2, not touching the outer border) near grid edge,
  each with a distinct number of enclosed holes.

Rule:
- Each colored legend shape defines a mapping: hole count → color.
- For every grey sprite, count its holes (enclosed bg regions not touching the sprite's bounding-box border).
- Recolor the grey sprite to the matching legend color; erase grey distractors with unlisted hole counts.

Key insight: Shape identity is encoded by topology (number of holes), not geometry or size.
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

<reference_puzzle index="5" source="train" puzzle_id="c920a713" lines="87-102">
<description>
Untangle overlapping colored rectangular frames into concentric square frames ordered by their visible crossings.

Input structure
- The background contains several one-cell-thick axis-aligned rectangular frames, one color per frame.
- Frames overlap, and the visible color at an overlap records a directed crossing from the hidden frame to the visible frame.
- Each foreground color still has visible pixels on the perimeter of its full rectangular bounding box.

Rule
- Infer each color's full rectangular frame from the bounding box of its visible pixels.
- Add a directed edge A -> B when B is visible at a crossing of A and B's frame perimeters.
- Find the unique path that visits every frame color once, then render those colors as one-pixel-thick concentric square frames from outside to inside.

Key insight
- The output is not a crop of the input; the crossing graph gives a single frame order, and that order is redrawn as nested rings.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="train" puzzle_id="3d6c6e23" lines="10-27">
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

<reference_puzzle index="7" source="train" puzzle_id="13f06aa5" lines="36-51">
<description>
Extend each colored marker from its directional sprite to a grid edge, painting a dotted ray and the whole edge side.

Input structure:
- The background is the dominant color.
- One repeated body color forms small 8-connected directional sprites.
- Each marker is a singleton non-background, non-body color embedded in one sprite.

Rule:
- The body pixels around a marker lie mostly behind it; the opposite direction is the marker's ray direction.
- Paint every second pixel from the marker along that ray, then paint the entire boundary side reached by the ray.
- If two painted boundary sides meet at a corner, make that shared corner black.

Key insight:
- The same seven-pixel body sprite rotates to encode the direction; its local center-of-mass bias points away from the ray.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="eval" puzzle_id="31f7f899" lines="7-21">
<description>
Sort vertical bars by height and redistribute them across fixed colored slots.

Input structure:
- bg background; a horizontal base line of a single color spans the full grid width.
- Several monochrome vertical bars of distinct colors rise upward from the base line.

Rule:
- Rank bar shapes by height (shortest to tallest).
- Rank original x-positions (slots) left to right.
- Place the i-th shortest shape into the i-th leftmost slot, recoloring it to that slot's original color.
- The base line is preserved unchanged.

Key insight: Each slot keeps its x-position and color identity; only the bar silhouettes are redistributed among slots by height order.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="train" puzzle_id="5b37cb25" lines="59-73">
<description>
Fill each plus-shaped background notch with the color of the frame side it opens toward.

Input structure
- A one-cell-thick rectangular frame has four solid-colored sides and neutral corner pixels.
- The frame interior contains a dominant background color and one foreground object color.
- Some object boundaries contain five-pixel plus-shaped background notches.

Rule
- Find every background plus whose opposite side is supported by foreground object pixels and whose open side faces one frame side.
- Recolor the five pixels of that plus with the color of the facing frame side.
- Leave all other pixels unchanged.

Key insight
The frame side colors are direction labels for the open notches in the foreground object.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="train" puzzle_id="7e4d4f7c" lines="9-24">
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

<reference_puzzle index="11" source="eval" puzzle_id="62593bfd" lines="32-46">
<description>
Sort objects to top or bottom edge by whether their shape (up to rotation/flip) is shared.

Input structure:
- bg = most common color.
- Several multicolor connected components (8-connected) scattered across the grid.

Rule:
- Compute each object's canonical shape (binary mask, invariant to rotation and flip).
- Objects whose shape appears in two or more components → translate to TOP edge (row 0).
- Objects whose shape is unique → translate to BOTTOM edge (row H−h).
- Horizontal position (left column) is preserved; only vertical translation changes.

Key insight: The sorting criterion is shape frequency (rotation-invariant), not area — duplicated shapes go up, unique shapes go down.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="train" puzzle_id="5792cb4d" lines="44-59">
<description>
Reverse the sequence of colors along the single 4-connected foreground path while keeping the path shape fixed.

Input structure:
- The background is the dominant color.
- All non-background pixels form one multicolor 4-connected one-pixel-wide path.
- The path has exactly two endpoints and no branches.

Rule:
- Walk the path from one endpoint to the other.
- Replace the colors on those same path cells by the original path colors in reverse order.
- Leave every background pixel unchanged.

Key insight:
- Nothing moves geometrically; only the color string written on the path is reversed between the endpoints.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="eval" puzzle_id="db0c5428" lines="32-51">
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

<reference_puzzle index="14" source="eval" puzzle_id="9385bd28" lines="71-87">
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

<reference_puzzle index="15" source="train" puzzle_id="25e02866" lines="14-31">
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

<reference_puzzle index="16" source="train" puzzle_id="c6141b15" lines="84-95">
<description>
Swap a connector segment with repeated marker sprites by connecting the marker anchors and stamping the sprite at the segment endpoints.

Input structure
The grid has one background color and exactly two foreground colors. One foreground color forms a single straight 8-connected segment. The other foreground color forms two or more identical centered sprites.

Rule
Build a background canvas of the same size. Draw straight connector-color segments between every pair of original sprite anchors. Then stamp the marker sprite, using its centered anchor, onto the two endpoints of the original connector segment.

Key insight
The original segment endpoints are the destinations for the repeated sprite, while the original sprite centers become the vertices of the new connector-color complete graph.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="train" puzzle_id="1b8318e3" lines="86-105">
<description>
Move each isolated colored pixel into the one-cell frame around a grey 2x2 block.

Input structure:
- Background is the dominant color.
- Grey foreground objects are solid 2x2 anchor blocks.
- Every other non-background, non-grey cell is a colored singleton source.

Rule:
- Keep the grey anchors fixed and clear all colored singleton source pixels.
- For each singleton and each anchor, the normal destination is the cell reached by
  clamping the pixel into the anchor's one-cell outer frame.
- If a diagonally outside singleton has an exact 8-direction sightline into an anchor,
  also allow the cell just before the first grey pixel on that ray.
- Choose distinct destinations that minimize total king-move distance; at equal
  distance prefer exact sightlines, then the shorter Manhattan movement.

Key insight: The grey 2x2 blocks define magnetic frames; conflicts are resolved
structurally by nearest-frame assignment rather than by color identity.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="train" puzzle_id="825aa9e9" lines="111-125">
<description>
Drop the moving-color objects downward until they rest above a one-row cushion before the obstacle or bottom edge.

Input structure:
- The grid has exactly three colors: empty background, a stationary obstacle/floor color, and a moving object color.
- The obstacle color is the unique non-background role that touches the bottom edge.
- Moving pixels start outside the obstacle cells, outside the row directly above each obstacle pixel, and outside the bottom row.

Rule:
- Keep the background and obstacle pixels fixed.
- Treat obstacle pixels, cells directly above obstacle pixels, and the bottom row as blocked.
- Repeatedly move every 4-connected moving-color object one row down when its translated shape does not hit blocked cells or another moving object.

Key insight: The visible floor has an invisible one-row cushion above it, so the objects fall under gravity but stop with one empty row before the floor or the grid bottom.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="train" puzzle_id="14b8e18c" lines="40-57">
<description>
Mark every complete square object with red pixels just outside its corners.

Input structure:
- A uniform background contains one or more monochrome 4-connected foreground objects.
- The relevant objects are square frames or filled squares: their bbox is square
  and all cells on the bbox border have the object's color.
- Non-square bars and incomplete fragments may also appear and are ignored.

Rule:
- Keep the input unchanged.
- For each complete square object, paint red on the two outside background
  pixels orthogonally adjacent to each of the square's four corners.

Key insight: The transformation is not based on size or color; it selects
objects whose bounding box is a complete square border and labels only their
exposed outer corner-adjacent cells.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="train" puzzle_id="6ffe8f07" lines="49-64">
<description>
Cast yellow horizontal and vertical rays from the teal rectangle until blue blockers stop them.

Input structure:
- A black-background grid containing one solid teal rectangle.
- Blue objects are blockers; other non-background objects may lie in the ray paths.
- Yellow is absent from the input and is the output paint color.

Rule:
- From every row of the teal rectangle, cast rays left and right to the grid edge.
- From every column of the teal rectangle, cast rays up and down to the grid edge.
- A ray stops before a blue pixel; all crossed non-blue, non-teal pixels become yellow.

Key insight:
The teal rectangle defines a full cross-shaped projection, while blue objects are walls and red/background cells are simply paintable space.
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
