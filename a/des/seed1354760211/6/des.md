## 第一步：分析与思考

### 参考题基础操作提取

从20道参考题中，可以提取出以下基础操作类别：

1. **位移/滑动**：对象按某种规则移动（题1按面积下移、题11矩形平移、题12向墙壁滑动）
2. **射线投射**：从种子点或区域沿某方向延伸射线（题2对角线反射、题19十字投射、题14标记推进）
3. **属性分离与重组**：将对象的不同属性（颜色、高度、形状）拆分后独立变换再组合（题7颜色与高度反向旋转、题8控制行列与载荷分离）
4. **模板匹配与填充**：按尺寸/特征匹配后嵌入（题15按孔洞数匹配、题20按尺寸嵌入、题4按调色板填充）
5. **轮廓/边界操作**：提取或变换轮廓（题10外轮廓、题3缺陷修补）
6. **编码解码**：网格中某些元素编码了控制信息（题6戳印地图、题9象限选择、题18多米诺方向编码）
7. **路径与拓扑**：沿连通路径遍历、展平（题13蛇形路径展平）
8. **模式传递**：从一个区域提取模式应用到另一个区域（题17标记模式迁移、题5块级周期扩展）

### 非显而易见的串联方式思考

- **面积编码指令**：一个对象的面积不仅可以控制位移量，还可以编码颜色索引、旋转角度或缩放因子——面积本身成为一种"隐藏通道"传递信息。
- **拓扑特征作为密钥**：对象的孔洞数量可以匹配模板，但更进一步，连通分量的端点数量、分支数量等拓扑特征都可以作为配对密钥。
- **空间关系编码变换类型**：两个对象之间的相对位置不仅可以指示滑动方向，还可以编码变换的**类型**本身（例如：在上方=旋转，在左方=翻转，在对角=转置）。
- **颜色的频率/排序作为操作序列**：颜色在网格中的出现频次可以编码操作的优先级或执行顺序。
- **对象的对称性缺陷指示操作**：一个近乎对称的对象，其打破对称性的像素位置和颜色可以同时编码"要做什么"和"用什么颜色做"。
- **边界行为作为规则提示**：物体到达网格边缘时的行为（裁剪、环绕、反射）本身可以由网格中其他元素的属性来决定。
- **几何量（周长、对角线长度、宽高比）的隐性排序**：在参考题中面积被用来控制位移，但周长、凸包面积等几何量同样可以提供隐藏排序或编码。
- **对象之间的空间拓扑（包围、相邻、共线）可以形成有向图**，图的结构本身编码了输出的排列。

### 头脑风暴方向

1. 利用对象的**对称性类型**（旋转对称阶数、镜像轴方向）作为隐藏参数控制变换。
2. 让对象的**凹陷开口方向**指示某种操作的方向。
3. 用前景对象之间的**曼哈顿距离关系**建立配对或排序。
4. 通过形状的**端点/分支数**作为离散参数选择变换。
5. 利用前景像素的**重心相对于网格中心的偏移**编码全局变换。
6. 对象与网格边界的**接触关系**作为变换参数。
7. 颜色频率在网格不同区域之间的**不对称分布**编码方向性信息。
8. 利用对象的**包围层级**（谁包围谁）决定输出的嵌套或覆盖顺序。

---

## 第二步：生成全新 ARC 谜题

<new_puzzle>
<description>
Detect the single axis of near-symmetry for each foreground shape, then mirror the defect pixels across that axis to complete perfect symmetry.

Input structure:
- The grid has a black background.
- Several monochromatic foreground shapes are scattered on the grid, each in a distinct color (e.g., red, green, blue, yellow, magenta).
- Each shape is "almost" reflectively symmetric about exactly one axis passing through its bounding-box center. The candidate axes are: horizontal (left-right mirror), vertical (top-bottom mirror), main diagonal (top-left to bottom-right), and anti-diagonal (top-right to bottom-left).
- "Almost symmetric" means exactly 1 to 3 pixels break the symmetry: they exist on one side of the axis but their mirror counterparts are background pixels.
- No two shapes overlap or touch each other. Each shape fits within a bounding box no larger than 9x9.

Rule:
- For each foreground shape, test all four candidate reflection axes through the center of its bounding box. The correct axis is the one that yields the fewest asymmetric pixels (i.e., pixels present on one side whose mirror position is background). There is always a unique best axis.
- For every pixel of the shape whose mirror position across the chosen axis is a background pixel, paint that mirror position with the shape's color.
- Pixels that already exist on both sides of the axis remain unchanged.
- The grid dimensions do not change. Shapes that are already perfectly symmetric remain untouched.

Key insight:
- The axis of near-symmetry is never drawn or marked; the solver must try all four candidate axes per shape and pick the best one. Different shapes in the same grid may use different axes. The task appears to be "complete a broken pattern" but the key difficulty is discovering which axis each shape was meant to be symmetric about.
</description>
</new_puzzle>

<new_puzzle>
<description>
Use each closed rectangular outline's perimeter pixel count as a rank index into a color legend bar, then fill that rectangle's interior with the indexed color.

Input structure:
- The grid background is azure.
- Several hollow rectangles of varying sizes are drawn with grey single-pixel-wide outlines on the grid. No two rectangles overlap or share any border pixel. Rectangle widths and heights range from 3 to 12.
- Each rectangle's interior is azure (background), making them appear empty.
- Along the very bottom row of the grid, a horizontal "color legend" is placed: a contiguous sequence of single colored pixels starting at column 0 and extending rightward. Each pixel in the legend is a distinct non-grey, non-azure color (e.g., red, green, yellow, blue, magenta, orange, maroon). The number of legend pixels equals the number of rectangles.

Rule:
- For each grey rectangle, count the number of grey pixels forming its border. For a rectangle whose outer dimensions are W columns wide and H rows tall, the perimeter pixel count is 2*W + 2*(H-2) = 2*W + 2*H - 4.
- Rank all rectangles by their perimeter pixel count in ascending order, assigning ranks 1, 2, 3, ... No two rectangles share the same perimeter.
- Read the color legend from left to right, yielding color_1, color_2, color_3, ...
- Fill the interior of the rectangle with rank k entirely with color_k. Interior means all pixels strictly inside the grey border.
- The grey outlines and the legend bar itself remain unchanged.

Key insight:
- The fill color is not determined by a rectangle's position, area, or aspect ratio, but by its perimeter pixel count ranked against all other rectangles. The solver must measure perimeters (not areas), sort them, and index into the separate color bar. The perimeter-based ranking is easy to overlook when areas visually differ more dramatically.
</description>
</new_puzzle>

<new_puzzle>
<description>
Each arrow-shaped sprite labels a cargo object by shared color; move each cargo in the reverse of the arrow's pointing direction until it docks against the arrow's tail.

Input structure:
- The grid has a black background.
- "Arrow sprites" are small 3-pixel objects shaped like an L (two pixels forming a straight shaft and one pixel at a 90-degree turn forming the arrowhead). The arrow points from the "tail" end (the free end of the shaft) through the shaft toward the "head" pixel (the pixel at the 90-degree bend). This gives exactly one of eight compass directions (N, NE, E, SE, S, SW, W, NW).
- Each arrow has a unique foreground color (e.g., red, blue, green, yellow, magenta).
- For each arrow, exactly one other connected foreground object of the same color exists elsewhere on the grid. This is the "cargo." Cargo objects are arbitrary connected shapes of 2-7 pixels.
- No two arrows share a color; each cargo matches exactly one arrow.

Rule:
- Determine each arrow's pointing direction from its pixel geometry: the direction vector goes from the tail pixel to the head pixel.
- Move the same-colored cargo object in the **opposite** direction of the arrow's point (i.e., toward where the tail side faces outward). Slide the cargo pixel by pixel in that direction until the cargo's leading edge is exactly one pixel away from (4-adjacent to) the arrow's tail pixel.
- Erase the cargo from its original position; redraw it at its new docked position.
- Arrow sprites remain in place and unchanged.
- Process arrows in top-to-bottom, left-to-right order of their head pixel. A cargo being moved treats previously settled cargos and all arrows as impassable obstacles and stops one cell before them.

Key insight:
- The arrow's 3-pixel L-shape hides a directional code: the solver must figure out which pixel is the head and which is the tail based on the L geometry, then reverse the direction. The shared color between arrow and cargo is the matching key, and the reversal of the arrow direction (pulling, not pushing) is the non-obvious rule.
</description>
</new_puzzle>

<new_puzzle>
<description>
Sort foreground objects by pixel count, then stack them centered inside a grey frame with the largest on the bottom and smallest on top, allowing smaller objects to overwrite larger ones where they overlap.

Input structure:
- The grid has a black background.
- A single hollow grey rectangle (the "frame") is drawn somewhere on the grid. Its interior is filled with background (black). The frame's interior is at least 10x10.
- Several foreground objects of distinct colors (non-grey, non-black) are placed elsewhere on the grid. Each is a connected group of same-colored pixels fitting within a bounding box no larger than 8x8. Objects do not overlap each other or the frame in the input.
- There are between 3 and 6 foreground objects.

Rule:
- For each foreground object, compute its bounding box and its pixel count (number of colored pixels).
- Extract each object within its bounding box (background pixels become transparent).
- Center each object's bounding box within the grey frame's interior. Centering is done by placing the bounding box so that the top and left margins are each floor((frame_interior_height - bbox_height) / 2) and floor((frame_interior_width - bbox_width) / 2).
- Composite the objects onto the frame's interior in **descending** order of pixel count: the object with the most pixels is drawn first (becomes the bottom layer), the next is drawn on top, and so on. The object with the fewest pixels is drawn last and appears fully visible on top.
- Erase the original foreground objects from their input locations. The output grid shows only the grey frame with the composited stack inside it. The grey border is preserved.

Key insight:
- The stacking order is determined purely by pixel count (area), not by spatial position, color, or bounding-box size. Smaller objects end up on top, which is visually counter-intuitive. The solver must realize that area-based sorting and centered overlay produce the output, and that the frame acts as the sole output viewport.
</description>
</new_puzzle>

<new_puzzle>
<description>
Each foreground object casts a diagonal shadow whose direction is encoded by a single marker pixel's grid-quadrant position, and whose length equals the object's own pixel count.

Input structure:
- The grid has a black background.
- Exactly one red single pixel (the "sun marker") is placed somewhere on the grid. It is isolated (not part of any other object).
- Several foreground objects of distinct colors (none red or black) are scattered on the grid, each a connected shape of 2-8 pixels. Objects do not touch each other or the red marker.

Rule:
- Determine the shadow direction from the red marker's quadrant relative to the grid center. Divide the grid into four quadrants using the center row (floor(H/2)) and center column (floor(W/2)):
  - Marker in top-left quadrant (row < center_row AND col < center_col): shadows go toward bottom-right (SE), i.e., each shadow pixel steps +1 row, +1 col.
  - Marker in top-right quadrant (row < center_row AND col >= center_col): shadows go toward bottom-left (SW), i.e., +1 row, -1 col.
  - Marker in bottom-left quadrant (row >= center_row AND col < center_col): shadows go toward top-right (NE), i.e., -1 row, +1 col.
  - Marker in bottom-right quadrant (row >= center_row AND col >= center_col): shadows go toward top-left (NW), i.e., -1 row, -1 col.
- For each foreground object, compute A = its pixel count.
- From every pixel of the object, walk A steps in the shadow direction. Paint each visited cell grey, but skip any cell that already contains a non-black, non-grey color (do not overwrite existing objects or the red marker). Also skip cells outside the grid boundary.
- The original objects and red marker remain unchanged in the output.

Key insight:
- The red marker serves a purely positional role: its quadrant encodes one global diagonal direction for all shadows, acting like a "sun" whose position determines where light comes from. Each object's own area determines how far its shadow extends, linking object size to shadow length. The solver must connect the marker's quadrant to the shadow direction and discover that the shadow length equals the object's pixel count.
</description>
</new_puzzle>

<new_puzzle>
<description>
For each pair of same-colored objects, draw a filled rectangle spanning between them in a third color determined by the pair's shared color and a color-mapping key object on the grid.

Input structure:
- The grid has a black background.
- A small rectangular "key block" sits in one corner of the grid. The key block is a K x 2 arrangement (K rows, 2 columns): the left column contains K distinct colors, and the right column contains K corresponding "bridge colors." Each row maps one color to its bridge color.
- Elsewhere on the grid, foreground objects appear in pairs: for each non-black color used outside the key block, exactly two separate connected objects of that color exist.
- Paired objects are separated by empty (black) space. No two differently-colored pairs share the same row span and column span simultaneously.

Rule:
- Read the key block as a color mapping: left-column color -> right-column bridge color.
- For each color that appears as a pair of objects, find the two objects and compute the smallest axis-aligned rectangle that contains both objects' bounding boxes (the union bounding box).
- Fill every black pixel inside this union bounding box with the bridge color looked up from the key. Do not overwrite any existing foreground pixels.
- The key block and all original objects remain unchanged.

Key insight:
- The pairs are linked by color, and the bridge between them is not the pair's own color but a different color obtained from the key mapping. The solver must locate the key block (identifiable as the only rectangular 2-column structure in a corner), decode the mapping, and realize that the bridge fills the gap region between same-colored pairs. The geometric operation is straightforward once the color indirection is understood.
</description>
</new_puzzle>

<new_puzzle>
<description>
Count the number of distinct colors touching each grey cell's 4-neighborhood, then replace that grey cell with the color that appears in the neighborhood exactly once (the unique neighbor color).

Input structure:
- The grid uses a black background.
- Several solid colored regions (non-black, non-grey) are placed on the grid, forming an irregular mosaic with gaps between them.
- Grey pixels fill certain single-cell gaps or narrow seams between the colored regions. Each grey pixel is 4-adjacent to at least two non-black, non-grey colored pixels.
- No colored region contains grey pixels; grey appears only in the gaps/seams.

Rule:
- For each grey pixel, examine its four cardinal neighbors (up, down, left, right).
- Collect the non-black, non-grey colors among those neighbors.
- If one of those colors appears exactly once while all others appear more than once (or there are only two neighbor cells with different colors), recolor the grey pixel to the color that appears the fewest times among its neighbors.
- If there is a unique tie (two colors each appearing once), choose the color whose region has fewer total pixels on the entire grid (the globally rarer color).
- Non-grey pixels are never changed. Grey pixels that have no non-black, non-grey neighbors remain grey.

Key insight:
- Each grey pixel acts as a "vote counter" for its immediate neighborhood. The recoloring rule is local (based on 4-neighbors), but tie-breaking requires a global view (total region sizes). The solver must first notice the local uniqueness rule, then discover the global tiebreaker from examples where ties occur. The grey pixels effectively become disputed boundary pixels claimed by the rarer adjacent region.
</description>
</new_puzzle>

<new_puzzle>
<description>
Rotate each object by 90 degrees clockwise a number of times equal to how many grid edges its bounding box touches, keeping it in place.

Input structure:
- The grid has an azure background.
- Several small foreground objects (each 3-8 pixels, each a distinct non-azure color) are placed on the grid. Objects do not overlap or touch each other.
- Some objects are in the grid interior (touching no grid edge), others have their bounding box touching exactly one, two, or three grid edges (top, bottom, left, or right). An object's bounding box "touches" an edge if any pixel of the object lies on that edge row or column.
- No object touches all four edges.

Rule:
- For each object, count T = the number of grid edges (top row 0, bottom row H-1, left column 0, right column W-1) that contain at least one pixel of the object.
- Rotate the object T times by 90 degrees clockwise around the center of its bounding box. Equivalently, rotate by (T * 90) degrees clockwise.
  - T=0: no rotation (object stays as-is).
  - T=1: 90 degrees clockwise.
  - T=2: 180 degrees.
  - T=3: 270 degrees clockwise (= 90 degrees counter-clockwise).
- After rotation, each pixel's new position is computed relative to the bounding-box center: for a 90-degree CW rotation, pixel at offset (dr, dc) from center moves to (dc, -dr) from center, rounded to nearest integer.
- Erase the object from its original pixels and draw the rotated version. If any rotated pixel falls outside the grid, that pixel is discarded (clipped).
- The object's color is preserved.

Key insight:
- The number of grid edges touched by each object's bounding box is the hidden rotation parameter. Objects in the interior remain fixed; objects along one edge rotate 90 degrees; corner objects rotate 180 degrees; objects spanning three edges rotate 270 degrees. The correlation between edge contact and rotation amount is the core discovery. Position near the boundary is not merely cosmetic — it actively determines the transformation.
</description>
</new_puzzle>
