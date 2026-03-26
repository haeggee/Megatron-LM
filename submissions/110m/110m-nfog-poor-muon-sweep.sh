# bash submissions/110m/110m-nfog-poor-muon.sh --muon-scale unit_rms_norm --lr 0.002 --mlr 2
# bash submissions/110m/110m-nfog-poor-muon.sh --muon-scale unit_rms_norm --lr 0.004 --mlr 2


bash submissions/110m/110m-nfog-poor-muon.sh --muon-scale unit_rms_norm --lr 0.002 --mlr 2
bash submissions/110m/110m-nfog-poor-muon.sh --muon-scale unit_rms_norm --lr 0.004 --mlr 2
bash submissions/110m/110m-nfog-poor-muon.sh --muon-scale unit_rms_norm --lr 0.002 --mlr 4
bash submissions/110m/110m-nfog-poor-muon.sh --muon-scale unit_rms_norm --lr 0.004 --mlr 4

bash submissions/110m/110m-nfog-poor-muon.sh --muon-scale unit_rms_norm --hs-embed --hs-embed-no-orthogonal --lr 0.002 --mlr 2
bash submissions/110m/110m-nfog-poor-muon.sh --muon-scale unit_rms_norm --hs-embed --hs-embed-no-orthogonal --lr 0.004 --mlr 2
bash submissions/110m/110m-nfog-poor-muon.sh --muon-scale unit_rms_norm --hs-embed --hs-embed-no-orthogonal --lr 0.002 --mlr 4
bash submissions/110m/110m-nfog-poor-muon.sh --muon-scale unit_rms_norm --hs-embed --hs-embed-no-orthogonal --lr 0.004 --mlr 4
# bash submissions/110m/110m-nfog-muon-embNO.sh --lr 0.004 --mlr 4 --logits-layer-scale-scale 1




bash submissions/110m/110m-nfog-poor-muon.sh --muon-scale none --lr 0.002 --mlr 2
bash submissions/110m/110m-nfog-poor-muon.sh --muon-scale none --lr 0.004 --mlr 2
bash submissions/110m/110m-nfog-poor-muon.sh --muon-scale none --lr 0.002 --mlr 4
bash submissions/110m/110m-nfog-poor-muon.sh --muon-scale none --lr 0.004 --mlr 4


bash submissions/110m/110m-nfog-poor-muon.sh --muon-scale none --hs-embed --hs-embed-no-orthogonal --lr 0.002 --mlr 2 --hs-u
bash submissions/110m/110m-nfog-poor-muon.sh --muon-scale none --hs-embed --hs-embed-no-orthogonal --lr 0.004 --mlr 2 --hs-u
bash submissions/110m/110m-nfog-poor-muon.sh --muon-scale none --hs-embed --hs-embed-no-orthogonal --lr 0.002 --mlr 4 --hs-u
bash submissions/110m/110m-nfog-poor-muon.sh --muon-scale none --hs-embed --hs-embed-no-orthogonal --lr 0.004 --mlr 4 --hs-u


# for mlr 4, use 0.001 to complete parabola
bash submissions/110m/110m-nfog-poor-muon.sh --muon-scale none --lr 0.001 --mlr 4
bash submissions/110m/110m-nfog-poor-muon.sh --muon-scale none --hs-embed --hs-embed-no-orthogonal --lr 0.001 --mlr 4 --hs-u

# for mlr 2, use 0.006 to complete parabola
bash submissions/110m/110m-nfog-poor-muon.sh --muon-scale none --lr 0.006 --mlr 2
bash submissions/110m/110m-nfog-poor-muon.sh --muon-scale none --hs-embed --hs-embed-no-orthogonal --lr 0.006 --mlr 2 --hs-u



# without hs u
bash submissions/110m/110m-nfog-poor-muon.sh --muon-scale none --hs-embed --hs-embed-no-orthogonal --lr 0.002 --mlr 2
bash submissions/110m/110m-nfog-poor-muon.sh --muon-scale none --hs-embed --hs-embed-no-orthogonal --lr 0.004 --mlr 2
bash submissions/110m/110m-nfog-poor-muon.sh --muon-scale none --hs-embed --hs-embed-no-orthogonal --lr 0.006 --mlr 2
bash submissions/110m/110m-nfog-poor-muon.sh --muon-scale none --hs-embed --hs-embed-no-orthogonal --lr 0.001 --mlr 4
bash submissions/110m/110m-nfog-poor-muon.sh --muon-scale none --hs-embed --hs-embed-no-orthogonal --lr 0.002 --mlr 4
bash submissions/110m/110m-nfog-poor-muon.sh --muon-scale none --hs-embed --hs-embed-no-orthogonal --lr 0.004 --mlr 4

