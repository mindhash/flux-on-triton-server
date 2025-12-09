# flux-on-triton-server



## Server
/opt/tritonserver/bin/tritonserver --model-repository /home/jovyan/flux-on-triton-server/model-repository/ --exit-on-error false


## Client 
python3 http_client.py --model flux1_service --prompt "butterfly in new york, 4k, realistic" --save-image --requests 1 --clients 2
