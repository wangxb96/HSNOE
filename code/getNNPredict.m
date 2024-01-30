function predict = getNNPredict(net,X)
    x = X';
    y = net(x);
    predict = vec2ind(y)';
    predict = predict - ones(size(predict,1), size(predict,2));
end