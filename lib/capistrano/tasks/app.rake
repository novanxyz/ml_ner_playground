namespace :app do
    desc 'Start Application'
    task :start_dev do
        on roles(:app) do
            execute "nohup /home/grumpycat/anaconda3/envs/monitoring/bin/python -u #{current_path}/ner_endpoint.py > #{shared_path}/logs/ner_endpoint.log 2>&1 &"
        end
    end

    desc 'Stop Application'
    task :stop_dev do
        on roles(:app) do
            execute "sudo pkill -f '/home/grumpycat/anaconda3/envs/monitoring/bin/python -u #{current_path}/ner_endpoint.py'"
        end
    end
end