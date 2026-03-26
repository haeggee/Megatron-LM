SQRT_MODELDIM=22.62  # approx sqrt(hidden_dim)

# bash submissions/110m/110m-nfog.sh --hs col
# bash submissions/110m/110m-nfog.sh --hs rowcol
# # bash submissions/110m/110m-nfog.sh --hs flat
# bash submissions/110m/110m-nfog.sh --hs flat --hs-r $SQRT_MODELDIM
# bash submissions/110m/110m-nfog.sh --hs flat --hs-r $SQRT_MODELDIM --lr 0.004
# bash submissions/110m/110m-nfog.sh --hs flat --hs-r $SQRT_MODELDIM --hs-u
# bash submissions/110m/110m-nfog.sh --hs flat --hs-r $SQRT_MODELDIM --hs-u --lr 0.004

# bash submissions/110m/110m-nfog.sh --hs-u --lr 0.001
# bash submissions/110m/110m-nfog.sh --hs-u --lr 0.004
# bash submissions/110m/110m-nfog.sh --hs-u --lr 0.008

# bash submissions/110m/110m-nfog.sh --hs rowcol --hs-u --lr 0.004
# bash submissions/110m/110m-nfog.sh --hs rowcol --hs-u



# # new
# bash submissions/110m/110m-nfog.sh --hs flat --hs-r $SQRT_MODELDIM --hs-u --lr 0.008
# bash submissions/110m/110m-nfog.sh --hs flat --hs-r $SQRT_MODELDIM --hs-u --lr 0.012
# bash submissions/110m/110m-nfog.sh --hs-u --lr 0.012
# bash submissions/110m/110m-nfog.sh --hs-u --lr 0.016

# # yet more

# bash submissions/110m/110m-nfog.sh --hs flat --hs-r $SQRT_MODELDIM --lr 0.008
# bash submissions/110m/110m-nfog.sh --hs flat --hs-r $SQRT_MODELDIM --lr 0.012



# bash submissions/110m/110m-nfog.sh --hs rowcol --lr 0.004
# bash submissions/110m/110m-nfog.sh --hs rowcol --lr 0.008
# bash submissions/110m/110m-nfog.sh --hs rowcol --lr 0.012


# bash submissions/110m/110m-nfog.sh --hs-u --b2 0


bash submissions/110m/110m-nfog-no-embed.sh --hs col
bash submissions/110m/110m-nfog-no-embed.sh --hs rowcol
bash submissions/110m/110m-nfog-no-embed.sh --hs rowcol --lr 0.004
bash submissions/110m/110m-nfog-no-embed.sh --hs rowcol --hs-u
bash submissions/110m/110m-nfog-no-embed.sh --hs rowcol --hs-u --lr 0.004
# bash submissions/110m/110m-nfog.sh --hs flat
bash submissions/110m/110m-nfog-no-embed.sh --hs flat --hs-r 22.62
bash submissions/110m/110m-nfog-no-embed.sh --hs flat --hs-r 22.62 --lr 0.004
bash submissions/110m/110m-nfog-no-embed.sh --hs flat --hs-r 22.62 --hs-u
bash submissions/110m/110m-nfog-no-embed.sh --hs flat --hs-r 22.62 --hs-u --lr 0.004

bash submissions/110m/110m-nfog-no-embed.sh --hs-u --lr 0.004
bash submissions/110m/110m-nfog-no-embed.sh --hs-u --lr 0.008
bash submissions/110m/110m-nfog-no-embed.sh --hs-u --lr 0.002
bash submissions/110m/110m-nfog-no-embed.sh --hs-u --lr 0.012

