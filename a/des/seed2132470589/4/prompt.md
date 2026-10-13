你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="d93c6891" lines="84-101">
<description>
Move each gray bar into its adjacent orange rectangle as full stripes from the contacted edge.

Input structure
- The scene uses black outside space, yellow fill, gray one-cell-thick bars, and solid orange rectangles.
- Each gray bar 4-touches exactly one orange rectangle; some orange rectangles have no touching gray bar.
- Gray bars touching the same orange rectangle all have the same orientation.

Rule
- Recolor every original gray cell to yellow.
- For each orange rectangle touched by gray bars, count the gray cells assigned to that rectangle.
- Horizontal gray bars repaint that many cells as complete orange rows starting from the contacted top or bottom row.
- Vertical gray bars repaint that many cells as complete orange columns starting from the contacted left or right column.
- Orange rectangles with no touching gray bar remain unchanged.

Key insight
- The gray pixels are not copied by shape; their count is transferred into the neighboring orange rectangle as edge-aligned row or column stripes.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="train" puzzle_id="17829a00" lines="21-36">
<description>
Move each colored object vertically until it touches the matching top or bottom border bar.

Input structure:
- The background is the dominant color.
- The top row is a full-width bar in one foreground color and the bottom row is a full-width bar in another.
- All other foreground pixels are 8-connected monochrome objects whose color matches one of those bars.

Rule:
- Top-bar-colored objects move upward so their bbox top is row 1.
- Bottom-bar-colored objects move downward so their bbox bottom is the row above the bottom bar.
- A downward-moving one-column segment whose landing overlaps its source becomes the continuous vertical segment swept between them.

Key insight:
The horizontal bars are destination anchors: color determines which border an object slides toward, while columns and object shape are otherwise preserved.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="eval" puzzle_id="e8686506" lines="172-185">
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

<reference_puzzle index="4" source="eval" puzzle_id="8698868d" lines="9-23">
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

<reference_puzzle index="5" source="eval" puzzle_id="5dbc8537" lines="10-23">
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

<reference_puzzle index="6" source="eval" puzzle_id="269e22fb" lines="46-60">
<description>
Complete a visible crop into the shared 20x20 binary motif in its matching orientation.

Input structure:
- The grid is a two-color rectangular crop of one rotation/reflection of the same 20x20 motif.
- One color is the motif foreground and the other color is the motif background.

Rule:
- Try each input color as the foreground mask.
- Match that mask against every possible window in every dihedral orientation of the motif.
- Repaint the uniquely matched full 20x20 motif with the matched foreground/background colors.

Key insight:
- The examples are partial windows of one canonical binary pattern; the local crop is distinctive enough to identify the full orientation and offset.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="train" puzzle_id="fe45cba4" lines="72-89">
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

<reference_puzzle index="8" source="train" puzzle_id="fc10701f" lines="20-35">
<description>
Move the orange rectangle onto the maroon rectangle and mark gated corridor crossings in red.

Input structure
- The dominant background contains black obstacle segments.
- There is one solid orange rectangle and one solid maroon rectangle of the same size.
- The two rectangles share either their row span or their column span, so the move is straight and axis-aligned.

Rule
- Erase the original orange rectangle to the background.
- Replace the maroon rectangle with the orange rectangle.
- Between the two rectangles, paint red across the moving rectangle's width or height only at corridor cells flanked by black obstacle segments perpendicular to the movement.

Key insight
- The black segments define gates in the straight corridor; only those gated openings, not every swept background cell, become red.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="train" puzzle_id="3d588dc9" lines="95-109">
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

<reference_puzzle index="10" source="eval" puzzle_id="8b9c3697" lines="9-23">
<description>
Insert red components into interior sockets of large structures, leaving a black trail.

Input structure:
- bg = most common. Large non-bg, non-red structures with interior cavities.
- Red components outside structures.

Rule:
- Interior sockets: bg cells inside a structure's bbox, adjacent to structure pixels.
- For each structure, find the direction where rays from sockets hit a red component.
- Slide that component into the socket; paint traversed cells black.
- Unmatched reds vanish.

Key insight: Docking direction is inferred by ray-casting from interior sockets to the nearest red component.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="train" puzzle_id="b7256dcd" lines="18-32">
<description>
Transfer each colored marker to its adjacent neutral mask component, then erase the markers.

Input structure
- The background is the most common color.
- The neutral mask color is the unique most frequent foreground color.
- Other foreground colors are marker pixels that may touch neutral 4-connected mask components.
Rule
- Remove every marker pixel by recoloring it to the background.
- For each 4-connected neutral mask component, find marker colors touching it by edge adjacency.
- If a component touches one marker color, recolor the whole neutral component to that marker color.
- If a component touches no marker color, leave it neutral.
Key insight
- Marker pixels label neighboring neutral regions but are not part of the final colored objects.
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

<reference_puzzle index="13" source="train" puzzle_id="57edb29d" lines="88-103">
<description>
Extract the single plain rectangle and redraw it with the aligned rectangles' border-relative marker pattern.

Input structure:
- The boundary color is the background/separator.
- Foreground objects are dense rectangular regions; exactly one rectangle has only its base color.
- Other rectangles contain their base color plus marker pixels.

Rule:
- Crop the only plain rectangle; this crop is the output canvas.
- Use marked rectangles whose row span overlaps the plain rectangle's row span.
- Transfer their marker pixels by preserving each marker run's distance from the nearest rectangle edges.

Key insight:
- The marked rectangles are examples of an edge-relative pattern, so the missing plain rectangle receives the same pattern resized to its own dimensions.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="train" puzzle_id="522fdd07" lines="8-23">
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

<reference_puzzle index="15" source="train" puzzle_id="ad3b40cf" lines="72-87">
<description>
Mirror the smallest non-axis foreground color across the full straight axis.

Input structure
- The background is the dominant color.
- One foreground color forms exactly one full row, column, main diagonal, or anti-diagonal reflection axis.
- The remaining foreground colors include a unique least-populated color to copy.

Rule
- Preserve the input grid.
- Select every pixel of the unique smallest non-axis foreground color.
- Reflect those pixels across the full straight axis and paint the reflected positions with the same color.

Key insight
- The axis color marks the mirror line, while the color population chooses which set of objects is duplicated.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="eval" puzzle_id="b99e7126" lines="34-48">
<description>
Scale a minority cell's inner pattern up to the cell-grid level, completing a fractal overlay.

Input structure:
- Grid divided into NxN cells by separator lines of a single color.
- Most cells share an identical interior pattern (majority); a few have a distinct pattern (minority).

Rule:
- Extract the inner pixel pattern from the minority cell.
- Treat each pixel as a cell-sized stamp: active pixels → minority pattern, inactive → majority pattern.
- Position this scaled-up pattern on the cell grid so existing minority cells align with active pixels.
- Fill in all remaining active positions with the minority pattern.

Key insight: The minority cell's interior is a miniature blueprint of where minority cells should appear at the grid scale — fractal self-similarity.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="train" puzzle_id="c61be7dc" lines="42-57">
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

<reference_puzzle index="18" source="train" puzzle_id="2faf500b" lines="30-44">
<description>
Remove each fracture marker line and push the surrounding object pixels one cell away from it.

Input structure
- A background grid with 8-connected foreground objects made from a body color and a rarer marker color.
- Each object has a non-square bbox; the marker pixels form one complete fracture line across the shorter bbox axis.

Rule
- Erase all marker pixels.
- For a wide object, each row has one marker; body pixels left of it shift one cell left and body pixels right of it shift one cell right.
- For a tall object, each column has one marker; body pixels above it shift one cell up and body pixels below it shift one cell down.

Key insight
- The marker line is a crack through the object, so the two sides separate by one pixel perpendicular to the crack.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="train" puzzle_id="d255d7a7" lines="99-114">
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

<reference_puzzle index="20" source="train" puzzle_id="5587a8d0" lines="11-27">
<description>
Redraw foreground objects as nested square rings ordered by object size.

Input structure:
- The grid has one dominant background color.
- Every foreground object is a 4-connected monochromatic component.
- Foreground object areas are distinct.

Rule:
- Sort the foreground objects by decreasing pixel count.
- Create a square output with side length 2 * number_of_objects - 1.
- The largest object color fills the outer ring, each next object color fills
  the next inner ring, and the smallest object color fills the center pixel.

Key insight: The objects' original shapes do not get copied; their areas
determine only the outside-to-inside order of the nested square pattern.
</description>
</reference_puzzle>

你的任务：
以参考题为标杆，运用发散性思维进行创新扩展。请发挥创造力，构思全新且巧妙的推理谜题。用中文创造全新的 ARC 谜题描述。
创新方式包括但不限于：
**机制重组**：将多道参考题各自的核心机制提取出来，组合成一个新的多步管道，用到新的场景。
**机制变异**：取一道参考题的核心机制，以机制为基础进行创新并与其他多种机制进行创新重组，使解题路径发生质变而非量变。
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
   - **允许使用**（这些都是人类能心算的视觉量）：数像素 / 孔洞 / 对象 / 邻居（≤10 量级）、比大小、按某个可见属性排序、奇偶判断、取最大 / 最小、用一个计数值作为另一个等量几何量（如 N 个像素 → 长度 N 的射线、面积 N → 边长 N）、判断接触 / 相邻 / 包含、粗略中心位置（簇的视觉中点）、"距离"作为"几步"而不是公式。
   - **不建议使用**（这些是算法量而非视觉量，会迫使解题者依赖公式而非视觉直觉）：曼哈顿 / 欧氏 / Chebyshev 距离作为精确查表键、凸包面积、二进制位编码 / 位运算、模运算 / 余数、拓扑距离（BFS / 最短路 / 跳数）、高精度方向区分（>4 方向，如时钟 12 方向）、精确几何质心坐标、凸角 / 凹角计数。
3. 问题必须能从少量示例中推断出来（通常为 3 对训练样例 + 1 个测试样例）。设计时从人类解题者的视觉角度思考：规则涉及的所有关键要素（对象、关系、属性）必须在网格上肉眼可见，不应依赖人眼无法直接观察到的隐含信息或需要推算的隐藏属性。规则背后的逻辑本身可以有一定挑战性，但解题所需的所有线索必须能从网格中直接获取。
4. 生成的问题描述旨在促进后续用于创建题目网格的代码生成。因此，对输入 / 输出网格的描述以及规则总结必须极其详细；语言风格应易于理解；对于特定对象（如箭头），必须在描述中明确其绘制方式。
5. 描述谜题时避免使用 0 或 1 这类颜色编号；应使用颜色名称，例如 "red"。可用颜色如下：Black、Blue、Red、Green、Yellow、Grey、Magenta、Orange、Azure、Maroon。
6. 5 个新题的核心推理机制应明显不同——即从观察样例到归纳规则的认知路径不同，而不仅仅是场景、对象或颜色不同。
7. 题目难度应来自机制的隐蔽性而非步骤数。用参考题作为难度标尺——如果你生成的题一个样例就能猜到规则，说明机制太透明，需要加深隐蔽性。
8. 优先从参考题的机制出发进行重组或推演，而非从物理隐喻出发。物理隐喻容易导致大量题目归结为一种模式，例如"沿方向移动到阻挡处停止"。
9. 题目的难度优先来自规则的**结构层数**，而不是单一步骤的复杂度。关于"结构层数"的判定标准：**只有解题者需要主动推理才能完成的步骤才算一层**。肉眼一扫即可获得的信息（如"找到缺口在哪一侧"、"数出像素个数"、"识别哪个对象最大"）不构成独立的推理层——它们只是变换的前置输入，不增加认知难度。真正的多层结构是指每一层都需要解题者发现一个非显然的规律。例如：
   - ✗ 伪两层："找到缺口方向"→"把东西推到对面" — 第一步肉眼可见，实际只有一层推理。
   - ✓ 真两层："通过孔洞数量匹配颜色图例"→"用匹配到的颜色重绘对象" — 两步都需要非平凡的归纳。

   如果你发现自己写出的题只是"对每个对象，根据它的某个属性做某变换"的单步规则，请引入一个需要主动推理的前置步骤。**用堆叠 tie-break 规则或复杂数学公式来"加难度"是劣解；用多一层需要归纳的结构嵌套来"加难度"才是好解**。

生成后对生成的题执行以下四项审查（审查内容不需要保存到文件）：
1. 原创性
  新题不可与上述参考题相同或类似，不可只是替换颜色、方向、对象类型等表层要素。判断标准：新题的核心推理路径应与任何一道参考题不同。
2. 确定性
  规则完全确定，同一输入只能产生唯一输出，每一步操作都有明确定义。
3. 可实现性
  所有必要细节都已给出——对象如何生成、如何识别、如何变换、边界情况如何处理，读完描述后可以直接写代码。
4. ARC 合规性
  网格尺寸合理（≤30×30），使用颜色不超过 10 种，规则复杂度适中，人类能从 3-5 个样例中发现规律。
