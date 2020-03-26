## Attention based keyword spotting 

A tensorflow implementation of the [Attention-based End-to-End Models for Small-Footprint Keyword Spotting] (https://arxiv.org/abs/1803.10916)

### requirements

see [requirements.txt]

## Usage

* feature: generate fbank or other feature, save to tfrecord format (todo)
* training: run [train.sh](train.sh)
* inference: run [infer.sh](infer.sh)
* evaluate: (todo)

## Modify

Modify the original model to streaming model which can support random length
inputs. 

## Support
* model: gru & lstm
* loss: focal loss & cross entropy loss & kl loss

### Refernce

Changhao Shan, Junbo Zhang, Yujun Wang, Lei Xie Attention-based End-to-End
Models for Small-Footprint Keyword Spotting.

