This folder provides the necessary manifest to test a image labelling application.
A demo application can be found in the tensorflow library(https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/examples/label_image), 
follow their given documentation to compile the image labelling application using Bazel.
The steps to use as the manifest are as follows -
Generate the manifest using the manifest template by using the following command - 
```
graphene-manifest \
-Dlog_level=error \
-Darch_libdir=/lib/x86_64-linux-gnu \
label_image_2.manifest.template > label_image_2.manifest
```


Then Generate the sgx manifest by usign the command - 

...
sudo graphene-sgx-sign \
--key ../../Pal/src/host/Linux-SGX/signer/enclave-key.pem \
--output label_image_2.manifest.sgx \
--manifest label_image_2.manifest
...

Generate a token using the command - 
...
graphene-sgx-get-token \
--output label_image_2.token --sig label_image_2.sig
...
