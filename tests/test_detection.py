import base64
import unittest
from unittest.mock import patch
import numpy as np

from unoq_dsp.vision import SSDDecoder, prepare_image
from unoq_dsp import CompiledModel,ImageClassifier


class DetectionTests(unittest.TestCase):
    def setUp(self):
        anchors=np.array([[.5,.5,.4,.2],[.5,.5,.4,.2],[.1,.1,.1,.1]],dtype="<f4")
        self.task={"kind":"ssd_fixed_anchors","schema":1,"num_boxes":3,"num_classes":2,
                   "anchors_xywh_f32_base64":base64.b64encode(anchors.tobytes()).decode(),
                   "scales_yxhw":[1,1,1,1],"exponential_size":True,"sigmoid_scores":False,
                   "boxes_output":"boxes","scores_output":"scores","labels":["person","cat"]}
        self.outputs={"boxes":np.zeros((1,3,4),"f4"),
                      "scores":np.array([[[.9,.1],[.1,.85],[.7,.2]]],"f4")}

    def test_anchor_decoding_and_class_agnostic_nms(self):
        detections=SSDDecoder(self.task).decode(self.outputs)
        self.assertEqual([d.label for d in detections],["person","person"])
        np.testing.assert_allclose(detections[0].box,[.3,.4,.7,.6],atol=1e-7)
        np.testing.assert_allclose(detections[1].box,[.05,.05,.15,.15],atol=1e-7)

    def test_class_aware_nms_keeps_different_classes(self):
        detections=SSDDecoder(self.task).decode(self.outputs,class_aware=True)
        self.assertEqual([d.label for d in detections],["person","cat","person"])

    def test_threshold_and_count_limits(self):
        decoder=SSDDecoder(self.task)
        self.assertEqual(decoder.decode(self.outputs,score_threshold=.99),[])
        self.assertEqual(len(decoder.decode(self.outputs,max_results=1)),1)
        with self.assertRaises(ValueError):decoder.decode(self.outputs,iou_threshold=float("nan"))

    def test_nonfinite_outputs_rejected(self):
        self.outputs["boxes"][0,0,0]=float("nan")
        with self.assertRaises(ValueError):SSDDecoder(self.task).decode(self.outputs)

    def test_preprocessing_uses_explicit_quantization_and_normalization(self):
        image=np.array([[[0,127,255]]],dtype="u1")
        x,size=prepare_image(image,{"shape":(1,1,1,3),"dtype":np.dtype("i1"),
                                   "quantization":{"scale":1/255,"zero_point":-128}},
                             {"mean":[0],"std":[255]})
        np.testing.assert_array_equal(x,np.array([[[[-128,-1,127]]]],"i1"))
        self.assertEqual(size,(1,1))

    def test_preprocessing_lookup_is_exact_for_every_byte_and_rgb_channel(self):
        image=np.stack([np.arange(256,dtype='u1'),np.arange(255,-1,-1,dtype='u1'),np.roll(np.arange(256,dtype='u1'),79)],axis=-1)[None]
        for mean,std,scale,zero in [([127.5],[127.5],1/128,128),([11,127,99],[17,64,23],.031,13)]:
            spec={'shape':(1,1,256,3),'dtype':np.dtype('u1'),'quantization':{'scale':scale,'zero_point':zero}}
            actual,_=prepare_image(image,spec,{'mean':mean,'std':std})
            values=(image.astype('f4')-np.array(mean,'f4'))/np.array(std,'f4')/np.float32(scale)
            expected=np.clip(np.where(values>=0,np.floor(values+.5),np.ceil(values-.5))+np.float32(zero),0,255).astype('u1')[None]
            np.testing.assert_array_equal(actual,expected)

    def test_classifier_does_not_dequantize_float_output_twice(self):
        spec={'shape':(1,1,1,3),'dtype':np.dtype('i1'),'quantization':{'scale':1/255,'zero_point':-128}}
        output={'shape':(1,2),'dtype':np.dtype('i1'),'quantization':{'scale':1/256,'zero_point':-128},'dequantize':True}
        model=CompiledModel(b'',{'input':spec,'output':output,'outputs':[output]})
        with patch('unoq_dsp.vision.Session') as session:
            session.return_value.metadata=model.metadata
            session.return_value.run.return_value=np.array([[.125,.875]],'f4')
            with ImageClassifier(model,mean=[0],std=[255],labels=['a','b']) as classifier:
                result=classifier.classify(np.zeros((1,1,3),'u1'),top_k=1)
            self.assertEqual(result,[{'class_id':1,'label':'b','score':.875}])


if __name__=="__main__":unittest.main()
