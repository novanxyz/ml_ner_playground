namespace :setup do
    desc "install anago depependencies"
    task :anago do
        on roles(:app) do
            execute "cd anago-src"
            execute "python setup.py"
        end
    end


    desc 'Uploading settings.cfg'
    task :settings do
        on roles(:app) do
            upload! StringIO.new(File.read('settings.cfg')), "#{shared_path}/settings/settings.cfg"
        end
    end
end