---
layout: page
title: Posts
permalink: /post/
---

<!-- {% include posts.html %} -->

<h1>Items</h1>

<ul>
  {% for post in site.posts %}
    <li><a href="{{ post.url }}">{{ post.title }}</a>test</li>
  {% endfor %}
</ul>
