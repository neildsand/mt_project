---
layout: page
title: Discussion and Analysis
description: Information about the project, website, and links to the paper and SI
img: Tubulin_Infographic.png # Add image post (optional)
caption: https://commons.wikimedia.org/wiki/File:Tubulin_Infographic.jpg #[Tubulin Infographic]("https://commons.wikimedia.org/wiki/File:Tubulin_Infographic.jpg")
permalink: index.html
sidebar: true
---

---


# {{site.data.about.title}}
{{site.data.about.authors}}

{% for entry in site.data.about %}

{% if entry[0] != 'title' %}
{% if entry[0] != 'authors' %}
## {{entry[0]}}
{{entry[1]}}
{% endif %}
{% endif %}
{% endfor %}
