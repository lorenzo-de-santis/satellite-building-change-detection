# Final LEVIR-CD Comparison

| Model | Best epoch | Epochs ran | Best val F1 | Test F1 | Test IoU | Test Dice | Precision | Recall | Test loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| siamese_temporal_attention_unet | 63 | 69 | 0.9058 | 0.9019 | 0.8214 | 0.9019 | 0.9000 | 0.9039 | 0.1189 |
| unet | 39 | 51 | 0.8820 | 0.8840 | 0.7922 | 0.8840 | 0.8787 | 0.8895 | 0.1681 |
| simple_cnn | 120 | 139 | 0.6944 | 0.7093 | 0.5496 | 0.7093 | 0.5895 | 0.8903 | 0.5404 |

## Test F1 Ranking

1. siamese_temporal_attention_unet: 0.9019
2. unet: 0.8840
3. simple_cnn: 0.7093
