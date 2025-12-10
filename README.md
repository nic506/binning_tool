Script run order: (1) Merging -> (2) Edge detection -> (3) Binning<br><br>

Merging script overview:<br>
  - Dilates input ROIs to decrease gaps between boundaries<br>
  - Generates a binary PNG mask, merging nearby ROIs<br>
  - Erosion of mask to counteract dilation, restoring approx original size<br>
  - Converts PNG mask to single ROI<br><br>
  
Edge Detection script overview:<br>
  - Assumes that ROIs are rectangle-like polygons (4 rough vertices) and that binning is desired along longest edge and its opposite edge<br>
  - For each ROI anchor point:<br>
    - Measures angles at multiple distances along polygon perimeter<br>
    - Scores how often angles fall within "corner-like" range (e.g. 30–150°)<br>
    - Computes angle consistency (standard deviation)<br>
    - Clusters nearby anchor points, keeping the best-scoring most consistent one<br>
  - Selects the top 4 vertex candidates based on score and consistency<br>
  - Extracts the longest edge between candidate vertices and the edge opposite to it<br>
### AUTOMATED EDGE DETECTION (above) FAILS FOR EXTREME CASES, SO CURRENTLY DEVELOPING A BETTER USER INTERFACE TO MANUALLY SELECT VERTICES ###<br><br>

Binning script overview:<br>
  - Vertical binning: Intersects tangents from one edge to the other, spaced at a defined width in μm<br>
  - Vertical + Horizontal binning: Divides each vertical bin horizontally into an equal number of segments<br>
  - Horizontal binning: Combines vertical bins within the same horizontal band<br>
  - Outputs a .zip file containing all bin ROIs<br><br>
