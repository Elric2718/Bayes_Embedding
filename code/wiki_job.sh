nohup ./pipeline_wiki.sh "2,3" "1" "" "raw" "2" "wiki_TransE" "desc_doc2vec" "pair" > nohup5.out 2>&1 & 

nohup ./pipeline_wiki.sh "1,2,3,4,5" "1" "" "new" "2" "wiki_TransE" "desc_doc2vec" "pair" > nohup6.out 2>&1 & 
nohup ./pipeline_wiki.sh "1,2,3,4,5" "1" "" "new" "2" "wiki_TransE" "desc_doc2vec" "single" > nohup7.out 2>&1 &

nohup ./pipeline_wiki.sh "1,2,3,4,5" "1" "" "new" "2" "wiki_TransE" "desc_sent2vec" "pair" > nohup8.out 2>&1 &
nohup ./pipeline_wiki.sh "1,2,3,4,5" "1" "" "new" "2" "wiki_TransE" "desc_sent2vec" "single" > nohup9.out 2>&1 &

nohup ./pipeline_wiki.sh "2,3" "1" "" "raw" "3" "wiki_TransE" "pagelink_node2vec" "pair" > nohup10.out 2>&1 & 
nohup ./pipeline_wiki.sh "1,2,3,4,5" "1" "" "new" "3" "wiki_TransE" "pagelink_node2vec" "pair" > nohup11.out 2>&1 & 
nohup ./pipeline_wiki.sh "1,2,3,4,5" "1" "" "new" "3" "wiki_TransE" "pagelink_node2vec" "single" > nohup12.out 2>&1 &

nohup ./pipeline_wiki.sh "1,2,3,4,5" "1" "" "new" "3" "wiki_TransE" "pagelink_line" "pair" > nohup13.out 2>&1 &
nohup ./pipeline_wiki.sh "1,2,3,4,5" "1" "" "new" "3" "wiki_TransE" "pagelink_line" "single" > nohup14.out 2>&1 &

