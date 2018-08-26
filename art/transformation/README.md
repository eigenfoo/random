# Transformation

George Ho and Jacqueline Soffer

## Description

These images were made for a flip book for the Machine Learning and Art class
(with Professors San Keene and Ingrid Burrington). The prompt was to

> "... use machine learning methods to transform a piece of media (images,
> sound, video, objects, etc) from one form to another. This can be same medium
> (i.e. transform one image into a different image or try and make a text into
> a different text) or going from one medium to another (ie. transform an text
> into an audio work)."

We found a wonderful pre-trained deep neural network, DeepWarp, that takes a
close-up shot of a person, and outputs a video of them rolling or crossing their
eyes. "It would be funny to see the stereotyped, meme of a dour, unsmiling Peter
Cooper doing this", was pretty much our initial motivation.

To expand this idea further, we wanted to take these stereotyped black-and-white
pictures and colorize them. 

Image colorization is an inherently ambiguous machine learning task for the same
reason that it is an inherently creative and artistic endeavor - there is no
obvious "right" or "wrong" colorization. Almost _any_ colorization is as "good"
as any other. We were therefore interested in seeing how different colorizers
worked on the same images, and comparing them by making a flip book out of it.

This meant that our first medium would be black-and-white images (specifically,
the single black-and-white image of Peter Cooper), and the second medium would
be video, or more accurately "moving pictures".

As it turned out, a flip book was a wonderful choice for a second medium. The
flip book presented the pictures in exactly the same way they were colorized:
frame by frame. Having discrete pages lets the viewer isolate each image
individually, and take as long as they like to study two or three consecutive
images. A flip book is also engages the reader much more than a projected screen:
it forces you to really study the images once its in your hand.

## Technical Details

Looped videos were also uploaded
[here](https://www.flickr.com/photos/155778261@N04/albums/72157664214657357).

**[`algorithmia/`](https://demos.algorithmia.com/colorize-photos/)**

A cloud-hosted repository of AI applications. Exact details about the colorizer
are unknown.

This colorizer gave encouraging results - instead of the entire image being
tones of sepia, the background was sometimes delightfully blue.

**`cooper/`**

Original image of Peter Cooper, and resulting video and frames once fed through
[DeepWarp](http://sites.skoltech.ru/compvision/projects/deepwarp/), a neural
network with a novel deep architecture specifically for image colorization. The
frames were extracted from the video using the `ffmpeg` command line tool.

**[`kvfrans/`](http://color.kvfrans.com/)**

A conditional generative adversarial network implemented by Kevin Frans, trained
on anime from [Safebooru](https://safebooru.org/).

The codebase is excruciating to use, poorly documented and uses a deprecated
version of Tensorflow. For that reason, we manually uploaded each of the 20
Cooper portraits to his [online demo](http://color.kvfrans.com/draw), and
manually downloaded the colorized output. The colorizer also allows you to give
"hints" on how you would like the colorized image to come out. We colorized the
images twice: once with no hints, and once by hinting that Cooper's eyes should
be blue.

This colorizer gave interesting results - as expected, the images are very
"fuzzy", as the model was _meant_ to be used on manga. As the image of Peter
Cooper was not line art, the colorization is not what Frans had intended, but is
nevertheless intriguing.

**[`pokemon/`](https://github.com/cameronfabbri/Colorful-Image-Colorization)**

A convolutional neural network implemented by Cameron Fabbri, trained on
screenshots of Pokemon Silver, Crystal, and Diamond.

This colorizer gave extremely disappointing results: each and every single image
came out pitch black. I suspect something was wrong with the model
checkpoint.

**[`tinyclouds/`](http://tinyclouds.org/colorize/)**

A convolutional neural network (specifically, the [VGG16
model](https://gist.github.com/ksimonyan/211839e770f7b538e2d8)) implemented by
Ryan Dahl and trained on the [ILSVRC
2012](http://academictorrents.com/details/a306397ccf9c2ead27155983c254227c0fd938e2/tech)
dataset - well over 1.2 million images, taking up 147GB of storage!

This colorizer gave somewhat boring results - all images came out in tones of
sepia.
