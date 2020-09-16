# Text-Removal
Computer Vision project that seeks to successfully remove text from an image by covering the text areas in a convincing way.

Here is the current method of removing the text:
  First, the most recurrent RGB colour in the image is determined. This is done by keeping a tally of the colours in the image, starting from the top left corner, that are stored in a dictionary. If a color appears more than all the other colours or it's tally count is greater than the total number of pixels in the image divided by 8, then the color is assumed to be the background color. This is all performed in a thread, using the built-in multiprocessing library, as this process can be very time consuming.

  Next, the locations of the text are predicted . This is done using "frozen_east_text_detection.pb", It has proven to be very successful in locating areas that have text. 

  Finally, a rectangle, the text bounding box, is drawn around the text areas. The pixels within the text bounding box are changed to the same colour as that of the background   color which was found previously. Then the new image is displayed.

The main issue with this method is the determining of the background color. Many cases could present themselves where the color found to be the most recurrent is not actually the color of the area surrounding the text or of the background of the image as a whole.

Further research will be done to attempt to further improve this project.

