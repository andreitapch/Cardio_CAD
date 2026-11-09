"""Contours and ground truth -- everything that is not pixels.

Worth keeping in mind when reaching for these, and documented at length in
the package README: **none of them is a tracking ground truth in vivo.** MITT
is a speckle tracker with its own error, ``scar_struct`` holds one static
contour, and the vendor files hold strain values with no contours at all. Only
``images.simulated`` carries true correspondence.
"""
