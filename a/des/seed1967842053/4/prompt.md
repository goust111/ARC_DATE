你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="332202d5" lines="49-64">
<description>
Fill every row from the nearest marked horizontal bar, turning bar rows and color-boundary ties into separator rows.

Input structure
- A dominant background fills the grid except for full-width horizontal bars and one vertical marker column.
- Outside the marker column, every row is monochrome: either background or a source bar color.
- Background rows share one marker color in the marker column; source bar rows share the other.

Rule
- Each non-source row takes the color of the nearest source bar, with that row's marker-column pixel set to the source-marker color.
- If a row is exactly midway between nearest bars of different colors, make the whole row the source-marker color.
- Each original source bar row becomes a source-marker row with the marker-column pixel changed to the background-row marker color.

Key insight
- The marked bars are horizontal seeds for a one-dimensional nearest-seed partition over rows; equal-distance rows are separators only when the competing seed colors differ.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="eval" puzzle_id="2ba387bc" lines="6-21">
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

<reference_puzzle index="3" source="train" puzzle_id="8dab14c2" lines="120-133">
<description>
Complete each boundary tab or notch by adding the complementary feature on the opposite side of the foreground object.

Input structure
- The grid has exactly two colors: a dominant background and one foreground object.
- The foreground object is a thick rectilinear shape with one-pixel tabs and notches on its boundary.

Rule
- A foreground tab is a one-neighbor boundary pixel; project through its full row or column span and remove the opposite edge pixel to make a matching notch.
- A background notch is a one-pixel cut with three foreground 4-neighbors; project across the full row or column span and add a tab just outside the opposite edge.
- Notches immediately beside a tab source are not independent sources, because that gap belongs to the tab's local boundary shape.

Key insight
- The changed pixels are not color-specific noise; they are paired jigsaw-like boundary features on opposite sides of the same thick row or column cross-section.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="eval" puzzle_id="b5ca7ac4" lines="7-21">
<description>
Move bordered 5×5 squares to their designated grid edges by border color.

Input structure:
- bg = most common color.
- Azure (8) and red (2) bordered 5×5 squares with colored interiors, scattered across the grid.

Rule:
- Azure squares → left edge (col 0); red squares → right edge (col W−5).
- Each square keeps its original row.
- Process azure left-to-right, red right-to-left (nearest-to-edge first).
- If a square's rows overlap an already-placed edge square, place it one square-width inward instead.

Key insight: Near-edge-first ordering plus row-overlap determines which square sits at the edge vs. the second column.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="eval" puzzle_id="7b5033c1" lines="7-21">
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

<reference_puzzle index="6" source="train" puzzle_id="c61be7dc" lines="42-57">
<description>
Flatten the interior object into a centered line and move the guide bars inward to bracket it.

Input structure:
- The dominant color is the background.
- One non-background guide color touches the boundary and forms exactly two full parallel bars plus a split perpendicular center bar.
- The other non-background color is a single 8-connected interior object whose area determines the flattened line length.

Rule:
- If the guide pair is vertical, make a vertical object-colored line centered between those bars; otherwise use the analogous horizontal line.
- The line length equals the number of interior object pixels and is centered on the split perpendicular guide bar.
- Rebuild the guide color as the perpendicular center bar plus two full bars immediately beside the flattened object line.

Key insight:
- The guide bars supply only the orientation and center; the interior object's pixel count supplies the length of the final one-cell-thick line.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="eval" puzzle_id="3e6067c3" lines="7-20">
<description>
Connect framed squares via colored bridges following a control-pixel sequence.

Input structure:
- bg = background. Border color forms equal-sized framed squares with unique inner fill colors.
- Control row below squares: colored pixels read left-to-right encode visitation order.

Rule:
- Start at the square matching first control color c0.
- For each consecutive pair (ci, ci+1), find the ci+1-colored square aligned in the same row or column band, preferring nearest.
- Draw a bridge colored ci between the pair, spanning the gap with inner-square thickness.

Key insight: Row/column-band alignment plus proximity resolves ambiguity when inner colors repeat.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="eval" puzzle_id="d59b0160" lines="39-54">
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

<reference_puzzle index="9" source="eval" puzzle_id="cb2d8a2c" lines="13-26">
<description>
Extend a green path from one grid edge to the opposite, routing around red bars at a distance dictated by their blue pixel count.

Input structure:
- bg = azure (8). Red/blue bars (color 2/1) act as obstacles.
- A single green (3) pixel sits on a grid border — the path start.

Rule:
- Each bar's keep-out radius equals its number of blue pixels; the path must stay farther than that from every bar cell.
- The green pixel extends inward toward the opposite edge, turning at right angles to navigate around forbidden zones.
- After the path is drawn, all blue pixels become red.

Key insight: Blue pixels on a bar encode its danger radius — more blue means a wider no-go zone the path must respect.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="train" puzzle_id="d6e50e54" lines="8-25">
<description>
Move each maroon marker toward the blue rectangle and recolor the rectangle red.

Input structure
A background canvas contains one solid blue rectangular object and several isolated
maroon marker pixels. Each marker is horizontally or vertically aligned with the
rectangle and lies outside it.

Rule
Start with a background canvas, repaint the blue rectangle red, and move each
maroon marker straight toward the rectangle along its row or column. Exactly one
marker is strictly closer to the rectangle than all others and enters the
rectangle boundary; every other marker lands in the adjacent exterior cell just
outside its aligned side.

Key insight
The single entering marker is the unique global nearest marker to the rectangle,
not the nearest marker on each of the four sides.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="train" puzzle_id="9f669b64" lines="26-48">
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

<reference_puzzle index="12" source="train" puzzle_id="a2d730bd" lines="43-56">
<description>
Extend each solid foreground rectangle toward its same-color isolated pixels with capped one-cell paths, clearing each original marker pixel.

Input structure
- The background is the dominant color.
- Each foreground color has exactly one solid rectangular object and one or more isolated same-color marker pixels.
- Every marker is horizontally or vertically aligned with the rectangle bbox and lies outside it.
Rule
- For each marker, draw a one-cell-wide straight path from the adjacent side of the rectangle through one cell beyond the marker.
- Add a three-pixel perpendicular cap at the rectangle end and around the marker endpoint.
- Recolor the original marker pixel to background.
Key insight
- The isolated pixels are not copied; they mark the hollow endpoints of capped protrusions grown from their matching rectangle.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="eval" puzzle_id="4c416de3" lines="99-118">
<description>
Replicate a pre-drawn corner pattern to every other marker-bearing corner.

Input structure:
- bg = most frequent color. Hollow 0-frames (possibly partial L-shapes); each
  frame-corner cell is either 0 or a painted (non-bg, non-0) color.
- >= 1 'source' corner: the pattern is already drawn (one color, both outward
  cells and an inward-diagonal trail).
- Other corners: a single colored pixel on the inward diagonal at the pattern's
  deepest (k,k) offset; the pixel's color is that corner's target color.

Rule:
- Extract the canonical (TL-oriented) pattern from any source.
- For each non-source corner whose marker cell is colored, paint the same
  pattern rotated to that corner's orientation, using the marker's color.

Key insight: the pattern is symmetric about the corner's inward-diagonal axis
(canonical dr == dc), so it reflects cleanly across all four corner orientations.
Painting uses each target's OWN marker color, not the source color.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="train" puzzle_id="5751f35e" lines="29-43">
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

<reference_puzzle index="15" source="train" puzzle_id="9f41bd9c" lines="10-30">
<description>
Move the side-attached foreground object to the opposite wall, shear its lower rows inward, and paint the crossed floor cells.

Input structure:
- The top row is the upper background color and the bottom row is the
  floor color.
- A single non-background, non-floor object touches exactly one vertical
  grid edge and rests immediately above the first solid floor row.

Rule:
- Clear the source object to the upper background color.
- Repaint each object row on the opposite side: the first three relative
  rows keep the edge alignment, and each lower row shifts one more pixel
  inward from that side.
- On the first floor row, repaint the traversed side-to-object segment
  maroon, stopping just before the final outer bottom support pixel.

Key insight:
- The side of the source object gives the horizontal direction, while the
  row index inside the object gives the one-cell-per-row diagonal shear.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="train" puzzle_id="db615bd4" lines="111-130">
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

<reference_puzzle index="17" source="train" puzzle_id="7e4d4f7c" lines="9-24">
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

<reference_puzzle index="18" source="eval" puzzle_id="7b80bb43" lines="8-21">
<description>
Remove diagonal staircase noise and complete broken straight lines.

Input structure:
- bg = background; one foreground color fg forming axis-aligned lines.
- Some fg pixels form 45° diagonal chains (8-connected but 4-disconnected from lines).

Rule:
- Delete all diagonal-only fg pixels (those with no orthogonal fg neighbor, only diagonal ones).
- Fill short gaps in horizontal/vertical lines where a deleted diagonal was nearby.
- Prune remaining diagonal-adjacent leaf pixels near former diagonal sites.

Key insight: Diagonal pixels are hints marking where a straight line is broken; their span sets the maximum gap size to fill.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="train" puzzle_id="1478ab18" lines="35-43">
<description>
Paint the teal outline of the right triangle determined by the missing corner of the grey-marker square.

Input structure
The grid is an orange background with exactly four foreground marker pixels of one color. The markers have a square bounding box: three markers are bbox corners, and the fourth marker is inside that square.
Rule
Find the one corner of the marker bounding square that has no marker. That missing corner is the right-angle vertex of a right isosceles triangle; paint its two axis-aligned legs and the opposite diagonal in teal, leaving existing marker pixels unchanged.
Key insight
The present corner marker opposite the missing corner is a decoy; the three triangle vertices are the missing corner and the two marker corners adjacent to it.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="eval" puzzle_id="62593bfd" lines="32-46">
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

