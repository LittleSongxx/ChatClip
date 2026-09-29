# E2E 评测源视频来源与许可

真实媒体不入库：本目录下的视频文件已被 `.gitignore` 排除，仅本文件入库。
重新获取方式见各条目的 Commons 文件页；`v3` 下载后需转码为 mp4（ogv 不在
任务创建后缀白名单）：

```bash
ffmpeg -i v3-wiki-tutorial-ch3.ogv -c:v libx264 -crf 23 -c:a aac -b:a 128k \
  -movflags +faststart v3-wiki-tutorial-ch3.mp4
```

| 文件 | 时长 | 分辨率 | 内容（垂类用途） | 许可 | 来源 |
|---|---|---|---|---|---|
| v1-chinese-wikipedia-experience.webm | 16m13s | 1920x1080 | 中文单人口播/会议演讲（高光、短视频 Hook、封面任务） | CC BY-SA 3.0 | [Contribution Gamification in Wikimedia Projects - The Chinese Wikipedia Experience](https://commons.wikimedia.org/wiki/File:Contribution_Gamification_in_Wikimedia_Projects_-_The_Chinese_Wikipedia_Experience.webm) |
| v2-chenpeisi-interview.webm | 5m46s | 1280x720 | 中文双人访谈（内容检索、说话人、访谈剪辑任务） | CC BY 3.0 | [星访谈·专访陈佩斯](https://commons.wikimedia.org/wiki/File:2020%E5%B9%B411%E6%9C%887%E6%97%A5%E3%80%90%E6%98%9F%E8%AE%BF%E8%B0%88%E3%80%91%E4%B8%93%E8%AE%BF%E9%99%88%E4%BD%A9%E6%96%AF%EF%BC%9A%E5%B8%8C%E6%9C%9B%E5%96%9C%E5%89%A7%E8%89%BA%E6%9C%AF%E5%9B%9E%E5%88%B0%E6%9C%AC%E8%BA%AB.webm) |
| v3-wiki-tutorial-ch3.mp4 | 5m39s | 1920x1080 | 中文教程/演示（内容检索、图文任务） | CC BY-SA 3.0 | [中文維基百科教學頻道第三章](https://commons.wikimedia.org/wiki/File:%E4%B8%AD%E6%96%87%E7%B6%AD%E5%9F%BA%E7%99%BE%E7%A7%91%E6%95%99%E5%AD%B8%E9%A0%BB%E9%81%93%E7%AC%AC%E4%B8%89%E7%AB%A0.ogv)（原 ogv 转码） |
| v4-fengxiaogang-interview.webm | 4m53s | 640x360 | 中文双人访谈（链式二剪、画幅、字幕任务） | CC BY 4.0 | [星访谈·专访冯小刚](https://commons.wikimedia.org/wiki/File:2023%E5%B9%B44%E6%9C%887%E6%97%A5_%E3%80%90%E6%98%9F%E8%AE%BF%E8%B0%88%E3%80%91%E4%B8%93%E8%AE%BF%E5%86%AF%E5%B0%8F%E5%88%9A%EF%BC%9A%E5%88%9B%E4%BD%9C%E8%80%85%E9%9C%80%E8%A6%81%E6%9C%89%E7%94%9F%E6%B4%BB_%E4%B8%8D%E8%83%BD%E8%A2%AB%E2%80%9C%E7%BB%91%E6%9E%B6%E2%80%9D.webm) |

全部来自 Wikimedia Commons，语言均为普通话。评测报告引用数字时须注明
样本为本表四段素材（总时长约 32 分钟）。
