# Personal webpage

## To add collapsible boxes

I wanted to add collapsible elements, as in [ejmastnak's blog](https://www.ejmastnak.com/tutorials/vim-latex/intro/). 
* This corresponds to using `<details>` blocks in HTML. 
* So that the content inside is properly rendered by Jekyll, we add a plugin following [this blog post](http://movb.de/jekyll-details-support.html). Content is in `_plugins/details.rb`.
* CSS styling of distill is overridden in `_sass/_distill.scss`. To make my `<details>` compatible notably with the dark/light themes, I copied more or less the styles from the code blocks and added
  ```scss
    details {
      color: var(--global-text-color);
      background-color: var(--global-code-bg-color);
      margin-top: 0;
      padding: 8px 12px;
      position: relative;
      border-radius: 6px;
      display: block;
      margin-bottom: 20px;
      grid-column: text;
      overflow: auto;
      max-width: 100%;
      summary {
        color: var(--global-theme-color);
      }
    }
  ```

All this enables a nice view with 
```
{% details Collapsible text _here_ %}
Correctly rendered **Markdown** text.
{% enddetails %}
```

## TODO
* footnotes are not properly rendered in collapsible boxes
* Make footnote a Jekyll tag, so that markdown within (e.g. links) is properly rendered 
* Remove `<details>` content from estimated reading time