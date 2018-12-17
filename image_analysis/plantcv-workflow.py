#!/usr/bin/env python

import os
import json
import argparse
import numpy as np
from plantcv import plantcv as pcv
import skimage
from scipy import ndimage as ndi


def options():
    parser = argparse.ArgumentParser(description="Camelina PlantCV workflow.")
    parser.add_argument("-i", "--image", help="Input image file.", required=True)
    parser.add_argument("-d", "--debug", help="Turn on debug, prints intermediate images.", default=None)
    parser.add_argument("-w", "--writeimg", help="write out images.", default=False, action="store_true")
    parser.add_argument("-r", "--result", help="result file.", required=False)
    parser.add_argument("-o", "--outdir", help="Output directory for image files.", required=False)
    parser.add_argument("-n", "--bkg", help="JSON config file for background images.", required=True)
    parser.add_argument("-p", "--pdf", help="PDF from Naive-Bayes.", required=True)
    args = parser.parse_args()
    return args


def main():
    # create options object for argument parsing
    args = options()
    # set debug
    pcv.params.debug = args.debug

    # read in a background image for each zoom level
    config_file = open(args.bkg, 'r')
    config = json.load(config_file)
    config_file.close()
    if "z2500" in args.image:
        bkg_image = os.path.expanduser(config["z2500"])
    elif "z500" in args.image:
        bkg_image = os.path.expanduser(config["z500"])
    elif "z1" in args.image:
        bkg_image = os.path.expanduser(config["z1"])
    else:
        pcv.fatal_error("Image {0} has an unsupported zoom level.".format(args.image))

    # Set output file name
    outfile = False
    if args.writeimg:
        outfile = os.path.join(args.outdir, os.path.splitext(os.path.basename(args.image))[0])

    # read in image
    img, path, filename = pcv.readimage(filename=args.image)

    # read in a background image
    bkg, bkg_path, bkg_filename = pcv.readimage(filename=bkg_image)

    # Detect edges in the background image
    bkg_sat = pcv.rgb2gray_hsv(rgb_img=bkg, channel="s")
    bkg_edges = skimage.feature.canny(bkg_sat)
    pcv.params.device += 1
    if args.debug == "print":
        pcv.print_image(img=bkg_edges, filename=str(pcv.params.device) + '_background_edges.png')
    elif args.debug == "plot":
        pcv.plot_image(img=bkg_edges, cmap="gray")

    bkg_dil = pcv.dilate(gray_img=bkg_edges.astype(np.uint8), kernel=3, i=1)

    # Close contours
    bkg_edges_closed = ndi.binary_closing(bkg_dil)
    pcv.params.device += 1
    if args.debug == "print":
        pcv.print_image(img=bkg_edges_closed, filename=str(pcv.params.device) + '_closed_background_edges.png')
    elif args.debug == "plot":
        pcv.plot_image(img=bkg_edges_closed, cmap="gray")

    # Fill in closed contours in background
    bkg_fill_contours = ndi.binary_fill_holes(bkg_edges_closed)
    pcv.params.device += 1
    if args.debug == "print":
        pcv.print_image(img=bkg_fill_contours, filename=str(pcv.params.device) + '_filled_background_edges.png')
    elif args.debug == "plot":
        pcv.plot_image(img=bkg_fill_contours, cmap="gray")

    # naive bayes on image
    masks = pcv.naive_bayes_classifier(rgb_img=img, pdf_file=args.pdf)

    # remove very small noise
    cleaned = pcv.fill(bin_img=masks["plant"], size=2)

    # Find edges in the plant image
    sat = pcv.rgb2gray_hsv(rgb_img=img, channel="s")
    edges = skimage.feature.canny(sat)
    pcv.params.device += 1
    if args.debug == "print":
        pcv.print_image(img=edges, filename=str(pcv.params.device) + '_plant_edges.png')
    elif args.debug == "plot":
        pcv.plot_image(img=edges, cmap="gray")

    # Combine the plant edges and the filled background
    combined_bkg = pcv.logical_and(bin_img1=edges.astype(np.uint8) * 255,
                                   bin_img2=bkg_fill_contours.astype(np.uint8) * 255)

    # Remove edges that overlap the background region
    filtered = np.copy(edges)
    filtered[np.where(combined_bkg == 255)] = False
    pcv.params.device += 1
    if args.debug == "print":
        pcv.print_image(img=filtered, filename=str(pcv.params.device) + '_filtered_edges.png')
    elif args.debug == "plot":
        pcv.plot_image(img=filtered, cmap="gray")

    # Keep everything in the cleaned naive Bayes mask and the filtered edges
    combined = pcv.logical_or(bin_img1=cleaned, bin_img2=filtered.astype(np.uint8) * 255)

    # Fill in noise in the pot region
    if "z2500" in args.image:
        pot_region = combined[450:1400, 850:1550]
        cleaned_pot = pcv.fill(bin_img=pot_region, size=100)
        combined[450:1400, 850:1550] = cleaned_pot
    elif "z500" in args.image:
        pot_region = combined[740:1500, 1000:1450]
        cleaned_pot = pcv.fill(bin_img=pot_region, size=100)
        combined[740:1500, 1000:1450] = cleaned_pot
    elif "z1" in args.image:
        pot_region = combined[1370:1750, 1050:1420]
        cleaned_pot = pcv.fill(bin_img=pot_region, size=100)
        combined[1370:1750, 1050:1420] = cleaned_pot
    else:
        pcv.fatal_error("Image {0} has an unsupported zoom level.".format(args.image))

    # Close edges
    closed_features = ndi.binary_closing(combined, structure=np.ones((3, 3)))
    pcv.params.device += 1
    if args.debug == "print":
        pcv.print_image(img=closed_features, filename=str(pcv.params.device) + '_closed_features.png')
    elif args.debug == "plot":
        pcv.plot_image(img=closed_features, cmap="gray")

    # image blurring using median filter
    blurred_img = pcv.median_blur(gray_img=closed_features.astype(np.uint8) * 255, ksize=(3, 1))
    blurred_img = pcv.median_blur(gray_img=blurred_img, ksize=(1, 3))
    cleaned2 = pcv.fill(bin_img=blurred_img, size=200)

    # Find contours using the cleaned mask
    contours, contour_hierarchy = pcv.find_objects(img, np.copy(cleaned2.astype(np.uint8) * 255))

    # Define region of interest for contour filtering
    if "z2500" in args.image:
        x = 300
        y = 30
        h = 400
        w = 1850
    elif "z500" in args.image:
        x = 500
        y = 30
        h = 710
        w = 1450
    elif "z1" in args.image:
        x = 580
        y = 30
        h = 1340
        w = 1320
    else:
        pcv.fatal_error("Image {0} has an unsupported zoom level.".format(args.image))
    roi, roi_hierarchy = pcv.roi.rectangle(x=x, y=y, w=w, h=h, img=img)

    # Filter contours in the region of interest
    roi_objects, hierarchy, kept_mask, obj_area = pcv.roi_objects(img, 'partial', roi, roi_hierarchy, contours,
                                                                  contour_hierarchy)

    # Analyze only images with plants present
    if len(roi_objects) > 0:
        # Object combine kept objects
        plant_contour, plant_mask = pcv.object_composition(img=img, contours=roi_objects, hierarchy=hierarchy)

        if args.writeimg:
            pcv.print_image(img=plant_mask, filename=outfile + "_mask.png")

        # Find shape properties, output shape image
        shape_header, shape_data, shape_img = pcv.analyze_object(img=img, obj=plant_contour, mask=plant_mask,
                                                                 filename=outfile)

        # Set the top of pot position
        if "z2500" in args.image:
            line_position = 1600
        elif "z500" in args.image:
            line_position = 1310
        elif "z1" in args.image:
            line_position = 680
        else:
            pcv.fatal_error("Image {0} has an unsupported zoom level.".format(args.image))

        # Shape properties relative to user boundary line
        boundary_header, boundary_data, boundary_img = pcv.analyze_bound_horizontal(img=img, obj=plant_contour,
                                                                                    mask=plant_mask,
                                                                                    line_position=line_position,
                                                                                    filename=outfile)
        # Determine color properties: Histograms, Color Slices and Pseudocolored Images,
        # output color analyzed images
        color_header, color_data, color_img = pcv.analyze_color(rgb_img=img, mask=plant_mask, bins=256,
                                                                hist_plot_type=None,
                                                                pseudo_channel="v", pseudo_bkg="img", filename=outfile)
        # Output shape and color data
        result = open(args.result, "a")
        result.write('\t'.join(map(str, shape_header)) + "\n")
        result.write('\t'.join(map(str, shape_data)) + "\n")
        for row in shape_img:
            result.write('\t'.join(map(str, row)) + "\n")
        result.write('\t'.join(map(str, color_header)) + "\n")
        result.write('\t'.join(map(str, color_data)) + "\n")
        result.write('\t'.join(map(str, boundary_header)) + "\n")
        result.write('\t'.join(map(str, boundary_data)) + "\n")
        result.write('\t'.join(map(str, boundary_img)) + "\n")
        for row in color_img:
            result.write('\t'.join(map(str, row)) + "\n")
        result.close()


if __name__ == '__main__':
    main()
