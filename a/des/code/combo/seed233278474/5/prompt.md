你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="825aa9e9" lines="111-125">
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

<reference_puzzle index="2" source="train" puzzle_id="b71a7747" lines="10-23">
<description>
Extract the hidden two-color image by compacting the rows and columns of its complete mask.

Input structure
The grid contains one pair of colors whose pixels occupy every cell in the
cross product of their active rows and active columns; all other colors form
the surrounding filler or separators.
Rule
Find that unique color pair, keep exactly the rows and columns touched by
the pair, and return the induced subgrid.
Key insight
The target image is not chosen by color id; it is the only two-color mask
whose occupied pixels form a full rectangular row/column product after
deleting unused rows and columns.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="eval" puzzle_id="8f215267" lines="7-21">
<description>
Count same-color patterns per frame and represent the tally as dots inside.

Input structure:
- bg = most common color.
- Hollow rectangular frames in various colors.
- Small monochrome pattern sprites scattered outside frames.

Rule:
- Remove all small patterns from the grid.
- For each frame of color C, count how many patterns share color C.
- Draw that many dots along the frame's center row, spaced by 2, starting from the right.

Key insight: Frames act as tally displays — scattered same-color patterns are "absorbed" and their count is shown as interior dots.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="train" puzzle_id="80214e03" lines="48-64">
<description>
Extract the color of each noisy block in a row/column lattice and mirror the coarse grid horizontally.

Input structure:
- A background surrounds a rectangular lattice of monochrome foreground blocks.
- Consecutive rows in a lattice band share the same set of visible foreground colors.
- Consecutive columns in a lattice band share the same set of visible foreground colors.
- Each row-band/column-band intersection has one foreground color, with possible background holes.

Rule:
- Group non-empty rows and columns into consecutive bands by their foreground color sets.
- Replace each lattice cell by the single foreground color inside that row/column-band intersection.
- Return the resulting coarse color grid with the column order reversed.

Key insight:
- Whole-row and whole-column color sets reveal the hidden lattice even when neighboring cells share a color or contain holes.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="train" puzzle_id="8dab14c2" lines="120-133">
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

<reference_puzzle index="6" source="eval" puzzle_id="7ed72f31" lines="8-22">
<description>
Mirror colored sprites using red (2) markers as reflection axes.

Input structure:
- bg = most common color. Red (2) = marker color.
- Colored sprites appear near red markers.

Rule:
- Red line within a sprite (vertical or horizontal) → mirror payload across that axis.
- Single red pixel in a mixed sprite → point-reflect payload around it.
- Isolated red pixel (alone, no payload) → point-reflect each adjacent 8-connected color component.
- Originals remain; mirrored pixels overlay.

Key insight: Red marker geometry (line vs single pixel) determines reflection type; isolated vs embedded red determines what gets reflected.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="eval" puzzle_id="65b59efc" lines="9-24">
<description>
Decode a stamp map: read templates, markers, and palette from a tiled grid, then render stamps.

Input structure:
- Grid divided into K tile columns (each N wide) separated by 1-col dashed separators.
- Top band (rows 0..N-1): K template masks, one per tile column.
- Middle band (rows N+1..2N): marker pixels whose color picks a template.
- Bottom band (rows 2N+2..): K palette swatches giving the output color per template.
- Separator rows at rows N and 2N+1 (dashed pattern of sep color).

Rule:
- Each marker's local position (lr, lc) within its tile column determines which N×N output block to stamp.
- The marker's color identifies which template to use; the palette recolors the template.

Key insight: Marker color encodes template index; local position within tile column encodes stamp block.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="train" puzzle_id="30f42897" lines="51-66">
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

<reference_puzzle index="9" source="train" puzzle_id="1478ab18" lines="35-43">
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

<reference_puzzle index="10" source="eval" puzzle_id="195c6913" lines="37-52">
<description>
Simulate particles along a path with alternating turn direction at collisions.

Input structure:
- Top-left: 2x2 monochrome blocks define a repeating color cycle (left-to-right).
- Isolated single pixels of a cycle color: start markers adjacent to the path.
- A 2x2 monochrome block of a distinct color: end marker.

Rule:
- Each start marker launches a particle moving straight, painting cycle colors in order.
- On hitting a boundary, paint that cell with the end color and turn 90°.
- Turn direction alternates: odd collisions → left (CCW), even → right (CW).
- If the scheduled turn is blocked, the particle stops.

Key insight: The alternating left/right turn rule is the core mechanism; without it the path would be trivial.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="eval" puzzle_id="8e5c0c38" lines="9-21">
<description>
Clean each color layer by enforcing vertical mirror symmetry.

Input structure:
- bg = background. Several non-bg colors, each forming a nearly-symmetric pattern about some vertical axis.
- Some pixels in each color are "noise" that breaks the symmetry.

Rule:
- For each non-bg color, find the vertical axis that best explains its pixel layout.
- Remove any pixel that lacks a same-color mirror counterpart across that axis.

Key insight: Each color has its own independent symmetry axis; the task is per-color denoising, not whole-grid symmetry.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="train" puzzle_id="52df9849" lines="49-67">
<description>
Complete each partially hidden simple foreground shape by recoloring the occluding pixels in its footprint.

Input structure
- The grid has a dominant background and several solid foreground shapes.
- Some shapes are partially hidden by other foreground colors, but their full
  footprint is still indicated by a foreground-filled bbox or by a foreground-
  filled 45-degree diagonal line.

Rule
- For each foreground color, complete its full rectangular bbox when that bbox
  contains no background and includes other colors.
- Otherwise, complete its 45-degree diagonal line when all visible pixels lie
  on that line and the hidden line cells are foreground, not background.
- Recolor only those hidden footprint pixels; leave all other pixels unchanged.

Key insight
- The occluded cells are exactly the non-background cells inside a simple
  shape's implied footprint, so completing shapes is a local recoloring problem.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="eval" puzzle_id="5961cc34" lines="8-21">
<description>
Build a red ray network by activating blue blobs via directional green ports.

Input structure:
- bg = most common color. Large blue blobs with adjacent green pixels marking port directions.
- Start marker: yellow pixel atop a short red segment at grid bottom.

Rule:
- Extend a ray upward from start; on hitting a blue blob, recolor it red.
- Activated blobs emit thick rays through green ports until boundary or another blob.
- Cascade until no new blobs are reached.

Key insight: Green pixels encode ray direction and width; the network activates as a chain reaction from one start point.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="train" puzzle_id="5587a8d0" lines="11-27">
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

<reference_puzzle index="15" source="train" puzzle_id="f28a3cbb" lines="42-50">
<description>
Compress each colored ray pixel inward until it touches that color's 3x3 anchor block.
Input structure
- A uniform background grid with non-background colors.
- Each foreground color contains one solid 3x3 anchor block; all other pixels of that color lie on straight 8-neighbor rays aimed at that block.
Rule
- Keep anchor blocks fixed, erase loose foreground pixels, and for each ray pack its pixels into consecutive cells adjacent to the anchor along the same ray.
Key insight
- The scattered pixels are displaced pixels on visible row, column, or diagonal sightlines back to their same-colored 3x3 block.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="eval" puzzle_id="5dbc8537" lines="10-23">
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

<reference_puzzle index="17" source="train" puzzle_id="84ba50d3" lines="36-54">
<description>
Drop the blue sprites downward relative to the red divider, with vertical sprites passing through to the bottom.

Input structure:
- The background is the dominant color.
- One non-background color forms a single full-width horizontal divider row.
- The other non-background color forms 4-connected sprites above the divider.

Rule:
- Clear the original sprite pixels.
- A one-column sprite falls to the bottom of its column and deletes the divider
  pixel it crossed.
- Any wider sprite falls until its unique widest horizontal row is immediately
  above the divider row; if the moved sprite overlaps the divider, it covers it.

Key insight:
The red row is not a generic floor: only one-column sprites pass through to the
bottom, while wider sprites are registered by their widest horizontal segment.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="train" puzzle_id="b25e450b" lines="28-43">
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

<reference_puzzle index="19" source="train" puzzle_id="2e65ae53" lines="88-103">
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

<reference_puzzle index="20" source="train" puzzle_id="ba1aa698" lines="41-55">
<description>
Extrapolate the moving sprite from a horizontal sequence of shared-border frames into the next standalone frame.

Input structure
- The grid is a row of equal-width framed panels sharing their vertical border columns.
- Each panel has the same fill color and one identical interior sprite at the same local columns.
- The sprite's top row changes by a constant amount from panel to panel.

Rule
- Build one output panel with the same height, frame color, fill color, and sprite.
- Continue the sprite's diagonal motion one more panel to the right.
- Because the output is a standalone panel, advance by the full panel width, not the shared-border spacing.

Key insight
- The panels are consecutive frames of motion; the output is the next frame after the sequence.
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
