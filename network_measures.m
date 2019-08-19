path = '/Users/harangju/Developer/data/wiki/graphs/';
files = dir(path);
iter = 100;
mod = {};
rich = {};
cont = {};
obs = {};
topics = {};
for i = 1 : size(files)
    filename = files(i).name;
    if ~strcmp(filename(1), '.')
        disp(filename)
        load([path filename])
        X = X'; % python is row-to-column
        topics{i} = filename(1:end-4);
        mod{i} = zeros(iter,1);
        for j = 1 : iter
            [~, mod{i}(j)] = community_louvain(X); % bct
        end
        rich{i} = rich_club_bd(X); % BINARY, bct
        cont{i} = ave_control(X); % wu-yan
        obs{i} = ave_control(X'); % wu-yan
    end
end
clear i j filename X iter
%%
topics = topics(cellfun(@(x) ~isempty(x), topics));
mod = mod(cellfun(@(x) ~isempty(x), mod));
rich = rich(cellfun(@(x) ~isempty(x), rich));
cont = cont(cellfun(@(x) ~isempty(x), cont));
obs = obs(cellfun(@(x) ~isempty(x), obs));
%%
save([path 'metrics.mat'],'mod','rich','cont','obs','topics')