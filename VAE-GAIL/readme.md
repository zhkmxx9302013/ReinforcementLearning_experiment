## 实现VAE

### 网络结构(参考论文‘Roubust Imitation of Divserse Behaviours’)

现有资源train 不动原文网络，将原网络中的wavenet替换成mlp

* Decoder input (latent space -> mlp -> action net  + state net)

![](./pics/decoder_input.png)

* Decoder sample

  ![](./pics/decoder_sample.png)

* Encoder (bi lstm + mlp -> $\mu$ + $\sigma​$ )

  ![](./pics/encoder.png)

* Loss数据流

  ![](./pics/loss.png)