export CORPUS_PATH=../../../Data/Reddit/reddit_body.txt
export CORPUS_NAME=reddit
export WANTED_WORD1=trans
export OUTPUT_DIR=./collocation_results_1M_reddit/minority


python3 ./find_collocation_in_text_corpus.py \
--corpus_path ${CORPUS_PATH} \
--corpus_name ${CORPUS_NAME} \
--wanted_word1 ${WANTED_WORD1} \
--output_dir ${OUTPUT_DIR}
