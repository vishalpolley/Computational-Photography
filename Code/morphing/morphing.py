"""Morphing one image to another based on specified control points."""

import os
import datetime
import numpy as np
import cv2
from matplotlib.delaunay import delaunay  # for triangulation


# Constants/global variables
out_dir = "out"  # where to store output images, videos
video_suffix = ".mov"  # including extension
video_fourcc = cv2.cv.FOURCC('m', 'p', '4', 'v')  # video codec
video_fps = 15.0  # frame rate


# Utility functions
def drawPoints(img_out, pts, color1=(0, 0, 255), color2=(0, 255, 0)):
    for i in xrange(pts.shape[0]):
        cv2.circle(img_out, (int(pts[i, 0]), int(pts[i, 1])), 1, color1, -1)  # default: red dot
        cv2.circle(img_out, (int(pts[i, 0]), int(pts[i, 1])), 5, color2, 2)  # default: green circle


def drawTriangles(img_out, tris, pts, color=(0, 255, 0)):
    pts = np.int_(pts)  # need integer points
    # For eacht t: t[0], t[1], t[2] are the point indices of the triangle
    cv2.polylines(img_out, [pts[t, :] for t in tris], True, color, 2)


def blend(img1, img2, alpha):
    return np.uint8(alpha * np.float32(img1) + (1.0 - alpha) * np.float32(img2))


def normalizeImage(img):
    minVal, maxVal, _, _ = cv2.minMaxLoc(img)
    return (np.float32(img) - minVal) / (maxVal - minVal) if maxVal != minVal else img


# Warping and morphing functions
def warpTriMesh(img, pts_orig, pts, triangles, showTriangles=False):
    """Warp using triangle mesh interpolation (piece-wise affine transform)."""
    
    # For each triangle, compute affine transform, use it to warp source region and combine to create warped image
    img_size = img.shape[0:2]
    img_warped = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
    mask = np.zeros(img_size, dtype=np.uint8)
    for t in triangles:
        # Find original and current triangle points
        tri_orig = pts_orig[t]
        tri = pts[t]

        # Compute affine transform and apply it
        # TODO: Currently very inefficient as it transforms the whole image but only uses a small region (for each triangle)!
        M_aff = cv2.getAffineTransform(tri_orig, tri)
        img_aff = cv2.warpAffine(img, M_aff, (img_size[1], img_size[0]))

        # Mask out target triangle region and copy to warped image
        mask.fill(0)
        cv2.fillPoly(mask, [np.int_(tri)], 255)
        np.copyto(img_warped, img_aff, where=np.bool_(mask)[:, :, np.newaxis])

    return img_warped


def morph(img_src, img_dst, pts_src, pts_dst,
        num_frames=11, animate=True, save_video=False,
        show_points=True, show_triangles=True,
        color_src=(255, 0, 0), color_dst=(0, 0, 255), color_tri=(0, 255, 0)):
    """Morph two images, given corresponding points."""

    # Check arguments
    assert img_src.shape == img_dst.shape, "Image dimensions do not match"
    img_size = img_src.shape[0:2]

    assert pts_src.shape == pts_dst.shape, "No. of points/dimensions do not match"
    num_pts = pts_src.shape[0]

    # Add corner points that should remain the same
    corner_pts = np.float32(
        [[0, 0],
         [img_size[1] - 1, 0],
         [0, img_size[0] - 1],
         [img_size[1] - 1, img_size[0] - 1]])

    pts_src = np.vstack((corner_pts, pts_src))
    pts_dst = np.vstack((corner_pts, pts_dst))
    num_pts = pts_src.shape[0]  # update num_pts

    ## Compute Delaunay triangulation (outside loop for consistency, but may result in non-Delaunay triangles)
    ## Note: Points must be defined such that triangles cover the entire image area
    centers, edges, triangles, neighbors = delaunay(pts_dst[:, 0], pts_dst[:, 1])
    
    # Morph loop
    try:
        if save_video:
            video_filename = os.path.join(out_dir, "morph_{}{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), video_suffix))
            video_out = cv2.VideoWriter()
            if video_out.open(video_filename, video_fourcc, video_fps, (img_size[1], img_size[0])):
                print "morph(): Writing video to file: {}".format(video_filename)
            else:
                print >> stderr, "morph(): Unable to create video file: {}".format(video_filename)
                save_video = False

        for t in np.linspace(0.0, 1.0, num_frames):
            ## Interpolate to find points at time t
            pts = t * pts_dst + (1.0 - t) * pts_src
            color = np.uint8(t * np.float32(color_dst) + (1.0 - t) * np.float32(color_src)).tolist()

            ## Warp src
            img_src_warped = warpTriMesh(img_src, pts_src, pts, triangles)
            img_out = img_src_warped  # [debug]
            
            ## Warp dst
            img_dst_warped = warpTriMesh(img_dst, pts_dst, pts, triangles)
            img_out = img_dst_warped  # [debug]

            ## Alpha-blend with alpha_dst = t, alpha_src = 1 - t
            img_out = blend(img_dst_warped, img_src_warped, t)

            if show_triangles:
                drawTriangles(img_out, triangles, pts, color_tri)  # draw current delaunay triangulation

            if show_points:
                drawPoints(img_out, pts, color, color)  # draw points with blended color

            if animate:
                cv2.imshow("Morph", img_out)
                key = cv2.waitKey(20)
                if key != -1:
                    key = key & 0xff
                    if key == 0x1b:  # Esc to break when animating
                        break
            
            if save_video:
                video_out.write(img_out)
    except KeyboardInterrupt:
        pass  # Ctrl+C to break from terminal

    if save_video:
        video_out.release()
        video_out = None
        print "morph(): Done writing video."


# Main script
if __name__ == "__main__":
    # Read two images: We want to morph from src to dst
    img_src = cv2.imread("jaguar-cx75.png")
    img_dst = cv2.imread("jaguar-standing.png")

    # Define corresponding point pairs (x, y)
    pts_src = np.float32(
        [[64, 100],
         [80, 217],
         [285, 65],
         [200, 232],
         [375, 82],
         [310, 300],
         [534, 140],
         [540, 278]])

    pts_dst = np.float32(
        [[185, 145],
         [230, 340],
         [340, 100],
         [350, 248],
         [430, 30],
         [420, 346],
         [500, 150],
         [492, 346]])

    # Visualize points [debug]
    '''
    img_src_out = img_src.copy()
    img_dst_out = img_dst.copy()
    drawPoints(img_src_out, pts_src, (255, 0, 0), (255, 0, 0))
    drawPoints(img_dst_out, pts_dst, (0, 0, 255), (0, 0, 255))
    cv2.imshow("src", img_src_out)
    cv2.imshow("dst", img_dst_out)
    '''

    # Call morphing function with src, dst images and points, and optional parameters
    num_frames = 31
    #morph(img_src, img_dst, pts_src, pts_dst, num_frames)  # just preview, showing control points and triangles
    #morph(img_src, img_dst, pts_src, pts_dst, num_frames, save_video=True)  # save video, showing control points and triangles
    morph(img_src, img_dst, pts_src, pts_dst, num_frames, save_video=True, show_points=False, show_triangles=False)  # save clean video, hiding control points and triangles
    #cv2.waitKey(3000)  # wait for up to 3 secs.
