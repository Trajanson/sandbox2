```
./bonnet_model_compiler.par \
  --frozen_graph_path=/temp/bagnet32_without_last_layer/saved_model.pb \
  --output_graph_path=retrained_graph.binaryproto \
  --input_tensor_name=input0 \
  --output_tensor_names=0.6831689243077527 \
  --input_tensor_size=224
```

docker build -t basicmodel7 .
docker run -it basicmodel7 /bin/bash