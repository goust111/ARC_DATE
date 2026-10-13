你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="494ef9d7" lines="16-30">
<description>
Slide each blue/teal or yellow/orange row pair together while preserving its left-to-right order.

Input structure
- The grid has a black background.
- Each nonempty row contains exactly two distinct foreground pixels.
- Blue/teal and yellow/orange row pairs are the moving pairs; other color pairs are distractors.

Rule
- For each moving pair, keep the left pixel fixed and move the right pixel left until it touches the left pixel.
- Clear the moved pixel's original location, and leave non-moving rows unchanged.

Key insight
- Only same-hue-family pixel pairs attract horizontally; separated unrelated colors do not compact.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="train" puzzle_id="6350f1f4" lines="62-81">
<description>
Decode a block-grid image by swapping each sprite-coded cell with a solid cell and each blank cell with the sprite.

Input structure
The grid is partitioned by black separator rows and columns into equal-size
square blocks. Each block is either a copy of one two-color sprite or a
uniform blank block of the sprite's minority color; grey pixels are erased
cells that act as wildcards.

Rule
Infer the repeated sprite from fully visible mixed-color blocks. Replace a
block matching that sprite, ignoring grey erasures, with a solid block of
the sprite's majority color. Replace a uniform blank block with the sprite.
Separator rows and columns, including grey erasures on them, become black.

Key insight
The small sprite is a block-level code for one macro-cell state, and the
uniform minority-color block is the other state; the output renders the
opposite code in each macro-cell while cleaning erased pixels.
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

<reference_puzzle index="4" source="train" puzzle_id="b745798f" lines="18-33">
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

<reference_puzzle index="5" source="train" puzzle_id="da6e95e5" lines="70-81">
<description>
Extract the central object from the marker cluster whose markers have the same color.

Input structure
The grid has a dominant background and several clusters; each cluster has four
same-color corner marker objects around one central monochromatic object.
Rule
Find every central object enclosed by four aligned corner markers of one color,
then choose the unique one whose object color equals the marker color.
Key insight
The answer is not the largest or most frequent object; it is the center of the
self-colored cluster, cropped to that object's bounding box with background kept.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="train" puzzle_id="22806e14" lines="23-37">
<description>
Mark the center of each odd-sized solid square with the plus marker color, erasing the marker only when it is outside the marked-center field.

Input structure:
- The dominant color is background.
- Exactly two foreground colors are present: one unique 3x3 plus marker and one color forming separated solid square objects.

Rule:
- For every non-marker 4-connected object, assert it is a solid square.
- If the square side length is odd, recolor its center pixel to the plus marker color; even-sided squares stay unchanged.
- Preserve the plus marker when its center lies inside the bounding box of the odd-square centers; otherwise erase it to background.

Key insight:
- Odd-sided solid squares have a single visible center pixel, and the plus supplies the color used to identify those centers.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="train" puzzle_id="97c75046" lines="88-107">
<description>
Move the single marker to the endpoint reached by sliding along the obstacle contour.

Input structure:
- A dominant background color fills the canvas.
- One single-pixel marker sits away from a connected obstacle region of the
  remaining color.
- The marker has exactly one horizontal or vertical sightline to the obstacle.

Rule:
- Find the last background pixel before the obstacle along that sightline.
- From that pixel, try the two diagonal turns that continue around the
  obstacle boundary.
- Move the marker to the endpoint of the turn that remains 4-adjacent to the
  obstacle for the longest distance.

Key insight:
The marker is not placed by distance to the obstacle; it first points to the
obstacle, then follows the visible outside contour until that contour segment ends.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="train" puzzle_id="6a980be1" lines="46-61">
<description>
Project the framed periodic stripe pattern across the canvas and erase the unused frame cells.

Input structure
- A bg canvas with a one-cell-thick rectangular frame of one color.
- Inside the frame are two parallel full-interior marker bars of another color.
- A third color fills the region between the marker bars on periodically spaced rows or columns.

Rule
- Infer the periodic row or column mask from the third-color stripes inside the frame.
- Paint the frame color along those full rows or columns, extended to the grid boundary.
- Extend the marker bars across the full canvas and repaint the source-color stripe cells between them.

Key insight
- The outer frame is not preserved as a border; it supplies the color for the periodic stripes selected by the inner source pattern.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="train" puzzle_id="e3fe1151" lines="36-55">
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

<reference_puzzle index="10" source="eval" puzzle_id="269e22fb" lines="46-60">
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

<reference_puzzle index="11" source="train" puzzle_id="9f41bd9c" lines="10-30">
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

<reference_puzzle index="12" source="train" puzzle_id="7e2bad24" lines="91-106">
<description>
Extend the diagonal blue ray through the grid, reflecting it off red line-segment mirrors.

Input structure
- Black-background grid with one diagonal 8-connected blue ray seed.
- Red foreground objects, when present, are straight horizontal or vertical mirror segments.
- Other foreground bars are transparent to the ray and may be overwritten where the ray crosses them.

Rule
- Start from the ray seed endpoint whose outward diagonal extension remains inside the grid.
- Paint the ray continuously in that diagonal direction until it leaves the grid.
- If the next cell is a red mirror, do not paint it; reflect by flipping the row direction on a horizontal mirror or the column direction on a vertical mirror.

Key insight
- The non-red bars are decoys: only red bars reflect the diagonal path, while all other crossed cells become blue.
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

<reference_puzzle index="14" source="train" puzzle_id="689c358e" lines="72-88">
<description>
Mark each arrow-like sprite's pointed frame endpoint with its color and the opposite endpoint black.

Input structure:
- The grid has a one-pixel solid rectangular frame around a dominant interior background.
- Inside the frame are monochrome six-pixel sprites shaped as asymmetric T arrows.
- Each sprite has a length-4 bar and a length-3 crossbar one cell closer to the pointed end.

Rule:
- For each sprite, follow its bar axis from the shorter end to the frame.
- Paint that frame endpoint with the sprite color.
- Paint the opposite endpoint on the same row or column black.
- Leave the interior background, sprites, and all other frame pixels unchanged.

Key insight:
- The direction is encoded by the crossbar's off-center position: the nearer end is the pointing end.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="train" puzzle_id="8dab14c2" lines="120-133">
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

<reference_puzzle index="16" source="train" puzzle_id="3cd86f4f" lines="6-18">
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

<reference_puzzle index="17" source="train" puzzle_id="aaef0977" lines="25-41">
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

<reference_puzzle index="18" source="eval" puzzle_id="78332cb0" lines="10-23">
<description>
Reorder 5×5 blocks into a connected strip by matching edge foreground pixels.

Input structure:
- bg = background; sep = divider color (full row/column lines).
- Several 5×5 sprite blocks in a grid, separated by sep lines.

Rule:
- Two blocks "connect" when their facing edges share foreground pixels at the same offset (e.g., A's right edge ∩ B's left edge).
- Chain all blocks in the orientation (horizontal or vertical) that admits a complete sequence.
- Output is a 1×N strip of blocks in chain order, separated by sep lines.

Key insight: Edge-contact overlap determines both block ordering and output orientation.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="train" puzzle_id="df978a02" lines="57-72">
<description>
Let the largest center-pointing object grow outward while every other object loses its inward tip.

Input structure
- The grid has a single dominant background color and four 8-connected monochromatic foreground objects.
- The objects surround a central background gap and point toward it from cardinal directions.
- Exactly one object has the largest area.

Rule
- Find the central background gap by minimizing aggregate distance to the four objects.
- Recolor the nearest-to-center pixel of each non-largest object to background.
- Add a three-pixel cap in the largest object's color on the side away from the center, centered on the shared row or column.

Key insight
- The edit is a transfer of prominence around a common center: losers retract at their tips, and the unique largest object extends outward from the center.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="train" puzzle_id="305b1341" lines="22-37">
<description>
Expand each sparse color object into a filled rectangular region using its legend-paired fill color.

Input structure
A background-color grid has a two-column legend in columns 0 and 1, starting at row 0: each row gives
a source color in the left column and that source color's fill color in the right column.  Outside the
legend, the foreground pixels are sparse dot objects whose colors are exactly the legend source colors.

Rule
Clear the legend.  For each source color, take the bbox of all of its non-legend dots, expand that bbox
by one pixel in every direction, and fill background cells inside the expanded rectangle with the paired
legend fill color while preserving every original non-legend dot.

Key insight
The legend order is also the layer order: earlier legend rows are in front, so fill rectangles are painted
from the last legend row back to the first and higher-priority fills overwrite lower-priority fills.
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
