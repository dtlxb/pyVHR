这是一个视觉心率识别，但我不知道拿它干什么。
对着脸拍的话，5-10秒会识别到大概心率的存在与否。

置信度-脸部识别和方向也好了。
只要pyvhr-realtime，然后监听8765和8766就行了。
   const ws = new WebSocket("ws://127.0.0.1:8765");
   ws.onmessage = (event) => console.log("BPM消息:", JSON.parse(event.data));
   ws.onopen = () => console.log("已连接 BPM WebSocket");
   ws.onclose = () => console.log("连接关闭");